"""
tests/test_train.py

Two genuinely different kinds of tests here, and they verify different things
- see the module docstring in train.py for the full explanation:

REAL tests (build_bnb_config, build_lora_config, build_training_arguments,
save_run_log): use the actual transformers/peft/trl libraries. These prove
the config objects train.py builds actually have the correct values.

MOCKED tests (run_training): use fake model/tokenizer/trainer objects to
verify the ORCHESTRATION is correct - the right functions get called in the
right order with the right arguments. These do NOT prove training actually
works on a GPU. That can only be verified in Colab.
"""

import json
import shutil
from pathlib import Path

from config import DatasetConfig, LoRAConfig, LoRAPreset, RunConfig, TrainingConfig
from train import (
    TrainingResult,
    build_bnb_config,
    build_lora_config,
    build_training_arguments,
    run_training,
    save_run_log,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)
DUMMY_DATASET = FIXTURES_DIR / "train_dummy.jsonl"
DUMMY_DATASET.write_text('{"instruction": "hi", "output": "hello"}\n' * 5)


def _make_run_config(base_model_id="mistral-7b", **overrides) -> RunConfig:
    training = overrides.pop("training", None) or TrainingConfig(force_cpu=True)
    return RunConfig(
        run_name=overrides.pop("run_name", "test-run"),
        base_model_id=base_model_id,
        dataset=DatasetConfig(file_path=DUMMY_DATASET),
        training=training,
        **overrides,
    )


# ---------- REAL tests: actual transformers/peft/trl objects ----------

def test_bnb_config_uses_run_configs_quantization_setting():
    cfg = _make_run_config(lora=LoRAConfig(quantization="fp4"))
    bnb = build_bnb_config(cfg)
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "fp4"
    print("PASS: BitsAndBytesConfig reflects the run's quantization setting")


def test_lora_config_pulls_target_modules_from_registry_not_run_config():
    # phi-3-mini's registry entry has FUSED target modules - this is the
    # actual integration point proving model_registry.py's Phi-3 fix
    # reaches train.py correctly, not just re-testing model_registry.py in isolation.
    cfg = _make_run_config(base_model_id="phi-3-mini")
    lora_cfg = build_lora_config(cfg)
    assert "qkv_proj" in lora_cfg.target_modules
    assert "gate_up_proj" in lora_cfg.target_modules
    assert "q_proj" not in lora_cfg.target_modules
    print("PASS: LoraConfig correctly uses Phi-3's fused target_modules from the registry")


def test_lora_config_reflects_preset():
    cfg = _make_run_config(lora=LoRAConfig(preset=LoRAPreset.HIGHER_QUALITY))
    lora_cfg = build_lora_config(cfg)
    assert lora_cfg.r == 32 and lora_cfg.lora_alpha == 64
    print("PASS: LoraConfig reflects the HIGHER_QUALITY preset (r=32, alpha=64)")


def test_training_arguments_reflect_training_config():
    cfg = _make_run_config(
        training=TrainingConfig(num_train_epochs=7, learning_rate=1e-5, force_cpu=True)
    )
    args = build_training_arguments(cfg, output_dir=Path("/tmp/opentune_test_output"))
    assert args.num_train_epochs == 7
    assert args.learning_rate == 1e-5
    assert args.use_cpu is True
    print("PASS: SFTConfig reflects the run's TrainingConfig values (force_cpu=True for this sandbox)")


def test_save_run_log_writes_valid_json():
    result = TrainingResult(
        run_name="log-test", adapter_path="/fake/path", final_train_loss=0.42,
        final_eval_loss=0.5, num_steps_completed=100, stopped_early=False,
    )
    logs_dir = FIXTURES_DIR / "logs"
    log_path = save_run_log(_make_run_config(run_name="log-test"), result, logs_dir)
    data = json.loads(log_path.read_text())
    assert data["result"]["final_train_loss"] == 0.42
    assert data["config"]["run_name"] == "log-test"
    print("PASS: save_run_log writes a valid, complete JSON log")
    shutil.rmtree(logs_dir)


# ---------- MOCKED tests: orchestration order only, NOT real training ----------

class _FakeTrainOutput:
    training_loss = 0.31


class _FakeTrainerState:
    global_step = 42
    best_metric = 0.28


class _FakeTrainer:
    """Records what it was called with, so the test can assert on call order/args."""

    calls = []

    def __init__(self, model, args, train_dataset, eval_dataset, peft_config):
        # This check is here BECAUSE of a real bug: the first Colab GPU run
        # caught train_dataset being a plain list instead of a real
        # datasets.Dataset, which trl.SFTTrainer requires internally. The
        # original mocked test didn't catch it because nothing here checked
        # the type. It does now.
        from datasets import Dataset

        assert isinstance(train_dataset, Dataset), (
            f"train_dataset must be a real datasets.Dataset, got {type(train_dataset)}"
        )
        assert isinstance(eval_dataset, Dataset), (
            f"eval_dataset must be a real datasets.Dataset, got {type(eval_dataset)}"
        )
        _FakeTrainer.calls.append("__init__")
        self.state = _FakeTrainerState()

    def train(self, resume_from_checkpoint=None):
        _FakeTrainer.calls.append(f"train:resume={resume_from_checkpoint}")
        return _FakeTrainOutput()

    def save_model(self, path):
        _FakeTrainer.calls.append(f"save_model:{path}")


def _fake_model_loader(repo_id, quantization_config, device_map):
    _FakeTrainer.calls.append(f"load_model:{repo_id}")
    return "fake-model-object"


def _fake_tokenizer_loader(repo_id):
    _FakeTrainer.calls.append(f"load_tokenizer:{repo_id}")

    class _FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return " ".join(m["content"] for m in messages)

        def __call__(self, text, truncation, max_length, padding):
            return {"input_ids": [1] * 8, "attention_mask": [1] * 8}

    return _FakeTok()


def test_run_training_calls_things_in_the_correct_order():
    """
    This does NOT verify training works - it verifies the orchestration
    calls tokenizer -> model -> dataset -> trainer -> train -> save, in that
    order, with the model repo_id resolved correctly from the registry key.
    Real GPU verification happens separately, in Colab.
    """
    _FakeTrainer.calls = []
    cfg = _make_run_config(run_name="orchestration-test")
    logs_dir = FIXTURES_DIR / "logs_orchestration"

    result = run_training(
        cfg,
        checkpoints_dir=FIXTURES_DIR / "checkpoints",
        logs_dir=logs_dir,
        model_loader=_fake_model_loader,
        tokenizer_loader=_fake_tokenizer_loader,
        trainer_cls=_FakeTrainer,
    )

    calls = _FakeTrainer.calls
    assert calls[0] == "load_tokenizer:mistralai/Mistral-7B-Instruct-v0.3"
    assert calls[1] == "load_model:mistralai/Mistral-7B-Instruct-v0.3"
    assert calls[2] == "__init__"
    assert calls[3] == "train:resume=None"  # fresh run - no checkpoint existed yet
    assert calls[4].startswith("save_model:")
    assert result.final_train_loss == 0.31
    assert result.num_steps_completed == 42
    print("PASS: run_training orchestrates tokenizer->model->trainer->train->save in correct order")
    shutil.rmtree(logs_dir, ignore_errors=True)
    shutil.rmtree(FIXTURES_DIR / "checkpoints", ignore_errors=True)


def test_run_training_resumes_from_existing_checkpoint():
    """
    Proves the actual feature this was built for: if a checkpoint from a
    previous, interrupted run already exists for this run_name, training
    resumes from it instead of restarting at step 0.
    """
    _FakeTrainer.calls = []
    cfg = _make_run_config(run_name="resume-test")
    checkpoints_dir = FIXTURES_DIR / "checkpoints_resume"
    logs_dir = FIXTURES_DIR / "logs_resume"

    # Simulate a previous run having already gotten partway through and
    # saved a checkpoint - exactly what a Colab disconnect would leave behind.
    fake_checkpoint_dir = checkpoints_dir / "resume-test" / "checkpoint-75"
    fake_checkpoint_dir.mkdir(parents=True)
    # trainer_state.json is the real completeness marker (confirmed from
    # Trainer's actual source - it's the last file written per checkpoint),
    # which _find_last_complete_checkpoint now requires to trust a checkpoint.
    (fake_checkpoint_dir / "trainer_state.json").write_text("{}")

    run_training(
        cfg,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        model_loader=_fake_model_loader,
        tokenizer_loader=_fake_tokenizer_loader,
        trainer_cls=_FakeTrainer,
    )

    train_call = next(c for c in _FakeTrainer.calls if c.startswith("train:"))
    assert str(fake_checkpoint_dir) in train_call, (
        f"expected resume to use {fake_checkpoint_dir}, got: {train_call}"
    )
    print(f"PASS: existing checkpoint detected and passed to trainer.train() - {train_call}")
    shutil.rmtree(checkpoints_dir, ignore_errors=True)
    shutil.rmtree(logs_dir, ignore_errors=True)


def test_incomplete_newest_checkpoint_falls_back_to_last_complete_one():
    """
    The actual scenario this whole feature exists for: a disconnect happens
    mid-save on checkpoint-100, leaving it without trainer_state.json, while
    checkpoint-75 (saved earlier, fully) is still intact. Resume must use
    checkpoint-75, not crash trying to resume from the broken checkpoint-100,
    and not silently ignore that a checkpoint exists at all.
    """
    from train import _find_last_complete_checkpoint

    checkpoints_dir = FIXTURES_DIR / "checkpoints_partial"
    run_dir = checkpoints_dir / "partial-test"
    run_dir.mkdir(parents=True)

    complete = run_dir / "checkpoint-75"
    complete.mkdir()
    (complete / "trainer_state.json").write_text("{}")

    incomplete = run_dir / "checkpoint-100"
    incomplete.mkdir()
    (incomplete / "model.safetensors").write_bytes(b"partial")  # weights saved...
    # ...but trainer_state.json never got written - the disconnect happened here

    result = _find_last_complete_checkpoint(run_dir)
    assert result == str(complete), (
        f"expected fallback to {complete} (the last COMPLETE checkpoint), got {result}"
    )
    print("PASS: incomplete newest checkpoint (checkpoint-100) correctly skipped, "
          "fell back to last complete one (checkpoint-75)")
    shutil.rmtree(checkpoints_dir, ignore_errors=True)


def test_all_checkpoints_incomplete_falls_back_to_fresh_start():
    """If every checkpoint present is incomplete, resume must return None
    (fresh start) rather than crash or resume from a broken checkpoint."""
    from train import _find_last_complete_checkpoint

    checkpoints_dir = FIXTURES_DIR / "checkpoints_all_broken"
    run_dir = checkpoints_dir / "broken-test"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint-50").mkdir()  # no trainer_state.json in either

    result = _find_last_complete_checkpoint(run_dir)
    assert result is None, f"expected None (fresh start), got {result}"
    print("PASS: all-incomplete case correctly falls back to a fresh start, not a crash")
    shutil.rmtree(checkpoints_dir, ignore_errors=True)


if __name__ == "__main__":
    test_bnb_config_uses_run_configs_quantization_setting()
    test_lora_config_pulls_target_modules_from_registry_not_run_config()
    test_lora_config_reflects_preset()
    test_training_arguments_reflect_training_config()
    test_save_run_log_writes_valid_json()
    test_run_training_calls_things_in_the_correct_order()
    test_run_training_resumes_from_existing_checkpoint()
    test_incomplete_newest_checkpoint_falls_back_to_last_complete_one()
    test_all_checkpoints_incomplete_falls_back_to_fresh_start()
    print("\nAll train.py tests passed (config + orchestration logic - NOT a real GPU training run).")
