"""
train.py

QLoRA fine-tuning loop, per design doc Section 18 and the Fine-Tuning
Pipeline flowchart (Section 11).

Testability split (see tests/test_train.py for exactly what's covered):
  - build_bnb_config / build_lora_config / build_training_arguments:
    pure config-object construction. Genuinely tested with the real
    transformers/peft libraries - no GPU or bitsandbytes package needed
    just to construct these objects, confirmed empirically before writing
    this module.
  - run_training: the actual orchestration - loads real model weights,
    attaches LoRA, and calls the training loop. This CANNOT be verified
    without a GPU. Its call-order and argument-passing logic is tested here
    against mocked model/trainer objects; the real end-to-end run has to
    happen in Colab.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from config import RunConfig
from data_pipeline import build_dataset
from model_registry import get_model


@dataclass
class TrainingResult:
    """What a completed (or early-stopped) training run reports back."""

    run_name: str
    adapter_path: str
    final_train_loss: float
    final_eval_loss: float | None
    num_steps_completed: int
    stopped_early: bool


def _to_hf_dataset(tokenized_examples: list) -> "Dataset":
    """
    Convert data_pipeline's tokenized examples (plain dicts) into a real
    datasets.Dataset - what SFTTrainer actually requires internally
    (confirmed against its real signature: train_dataset is typed strictly
    as datasets.Dataset | IterableDataset, not list). This was the exact gap
    the Colab GPU smoke test caught - the mocked orchestration tests didn't
    catch it because the fake trainer never validated the type it received.

    Only input_ids/attention_mask are kept. data_pipeline's TokenizedExample
    also carries formatted_text for debugging/introspection - useful in
    data_pipeline's own tests, but not something SFTTrainer's default
    collator needs, and not worth risking on unverified collator behavior
    with an extra non-tensor string column.
    """
    from datasets import Dataset

    return Dataset.from_list(
        [
            {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]}
            for ex in tokenized_examples
        ]
    )


def build_bnb_config(run_config: RunConfig):
    """Construct the 4-bit quantization config from the run's LoRAConfig settings."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=run_config.lora.quantization.value,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(run_config: RunConfig):
    """
    Construct the peft.LoraConfig for this run.

    target_modules comes from the model registry, NOT from run_config
    directly - this is what makes the Phi-3 fused-projection fix in
    model_registry.py actually take effect at training time.
    """
    from peft import LoraConfig

    model_entry = get_model(run_config.base_model_id)
    return LoraConfig(
        r=run_config.lora.r,
        lora_alpha=run_config.lora.lora_alpha,
        lora_dropout=run_config.lora.lora_dropout,
        target_modules=model_entry.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_training_arguments(run_config: RunConfig, output_dir: Path):
    """Construct trl.SFTConfig (training loop hyperparameters) from the run's TrainingConfig."""
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=run_config.training.num_train_epochs,
        per_device_train_batch_size=run_config.training.per_device_train_batch_size,
        learning_rate=run_config.training.learning_rate,
        save_steps=run_config.training.checkpoint_every_n_steps,
        logging_steps=max(1, run_config.training.checkpoint_every_n_steps // 5),
        gradient_checkpointing=run_config.lora.gradient_checkpointing,
        seed=run_config.training.seed,
        max_length=run_config.dataset.max_sequence_length,
        use_cpu=run_config.training.force_cpu,
        bf16=not run_config.training.force_cpu,
        fp16=False,
    )


def save_run_log(run_config: RunConfig, result: TrainingResult, logs_dir: Path) -> Path:
    """
    Write the run's config + result as one JSON file under logs/<run_name>/.

    This is the reproducibility artifact from design doc Section 7/22 - pure
    file I/O, genuinely tested for real, no GPU involved.
    """
    run_log_dir = logs_dir / run_config.run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_log_dir / "run_log.json"
    log_path.write_text(
        json.dumps(
            {"config": json.loads(run_config.model_dump_json()), "result": asdict(result)},
            indent=2,
        )
    )
    return log_path


def _find_last_complete_checkpoint(output_dir: Path) -> str | None:
    """
    Like transformers.trainer_utils.get_last_checkpoint, but validates each
    candidate actually finished writing before trusting it - see the
    explanation in run_training for why this matters (trainer_state.json is
    the last file Trainer writes per checkpoint, so its absence means an
    interrupted save, not a real checkpoint).

    Walks backward from the newest checkpoint by step number until it finds
    one with trainer_state.json present, or returns None if none qualify -
    which correctly falls through to a fresh start rather than crashing on
    a checkpoint that only exists in name.
    """
    import re

    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    candidates = [
        (int(match.group(1)), p)
        for p in output_dir.iterdir()
        if p.is_dir() and (match := checkpoint_pattern.match(p.name))
    ]
    for _, path in sorted(candidates, key=lambda x: x[0], reverse=True):
        if (path / "trainer_state.json").exists():
            return str(path)
        print(f"Skipping {path} - looks incomplete (no trainer_state.json), likely an interrupted save.")
    return None


def run_training(
    run_config: RunConfig,
    checkpoints_dir: Path,
    logs_dir: Path,
    model_loader=None,
    tokenizer_loader=None,
    trainer_cls=None,
) -> TrainingResult:
    """
    Full orchestration: load model+tokenizer in 4-bit, attach LoRA, build the
    dataset, train, save the adapter, log the run.

    model_loader / tokenizer_loader / trainer_cls are injectable (default to
    the real transformers/trl classes when None) specifically so this
    function's ORDER OF OPERATIONS can be verified against fakes in tests,
    without needing a GPU or real model download. See tests/test_train.py -
    those tests confirm the orchestration logic is correct; they do not and
    cannot confirm that training actually converges on real weights. That
    verification has to happen in Colab, against this same function.
    """
    if model_loader is None:
        from transformers import AutoModelForCausalLM

        model_loader = AutoModelForCausalLM.from_pretrained
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer_loader = AutoTokenizer.from_pretrained
    if trainer_cls is None:
        from trl import SFTTrainer

        trainer_cls = SFTTrainer

    model_entry = get_model(run_config.base_model_id)
    output_dir = checkpoints_dir / run_config.run_name

    # Resume support: if this run_name has a checkpoint from a previous,
    # interrupted run, continue from there instead of starting at step 0.
    #
    # get_last_checkpoint() only validates a folder's NAME (checkpoint-N),
    # not its contents - confirmed by reading Trainer._save_checkpoint's
    # actual source, where trainer_state.json (the completeness marker) is
    # written LAST, after weights/optimizer/scheduler. A disconnect mid-save
    # can leave a checkpoint-N folder that looks valid by name but is
    # actually incomplete. _find_last_complete_checkpoint walks backward
    # from the newest checkpoint until it finds one that actually finished
    # writing, rather than trusting the newest name blindly.
    resume_from_checkpoint = None
    if output_dir.exists():
        resume_from_checkpoint = _find_last_complete_checkpoint(output_dir)
        if resume_from_checkpoint is not None:
            print(f"Found existing checkpoint at {resume_from_checkpoint} - resuming, not restarting from step 0.")

    tokenizer = tokenizer_loader(model_entry.repo_id)
    bnb_config = build_bnb_config(run_config)
    model = model_loader(model_entry.repo_id, quantization_config=bnb_config, device_map="auto")

    train_raw, val_raw = build_dataset(run_config.dataset, tokenizer)
    train_data = _to_hf_dataset(train_raw)
    val_data = _to_hf_dataset(val_raw)
    lora_config = build_lora_config(run_config)
    training_args = build_training_arguments(run_config, output_dir)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))

    result = TrainingResult(
        run_name=run_config.run_name,
        adapter_path=str(output_dir),
        final_train_loss=train_result.training_loss,
        final_eval_loss=getattr(trainer.state, "best_metric", None),
        num_steps_completed=trainer.state.global_step,
        stopped_early=False,  # updated once early-stopping callback is wired in
    )
    save_run_log(run_config, result, logs_dir)
    return result
