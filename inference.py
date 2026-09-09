"""
inference.py

Loading a saved adapter and generating chat responses, per design doc
Section 21 and the Inference Pipeline flowchart (Section 11).

Same testability split as train.py/evaluate.py: format_chat_prompt is pure
logic, tested for real. load_finetuned_model and generate_response need a
GPU and real weights - injectable loaders let the orchestration be tested
via mocks; real verification happens in Colab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_registry import get_model


@dataclass
class LoadedModel:
    """A base model with an attached adapter, ready for generation."""

    model: object
    tokenizer: object
    adapter_path: str


def format_chat_prompt(user_message: str, tokenizer) -> str:
    """Apply the tokenizer's chat template to a single user message, ready for generation."""
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_finetuned_model(
    base_model_id: str,
    adapter_path: Path,
    model_loader=None,
    tokenizer_loader=None,
    peft_loader=None,
) -> LoadedModel:
    """
    Load the base model fresh and attach a previously-saved LoRA adapter.

    Loaders are injectable so the call order (tokenizer -> base model ->
    attach adapter) is verifiable without a GPU - see tests/test_inference.py.
    Real weight loading needs your Colab GPU.
    """
    if model_loader is None:
        from transformers import AutoModelForCausalLM

        model_loader = AutoModelForCausalLM.from_pretrained
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer_loader = AutoTokenizer.from_pretrained
    if peft_loader is None:
        from peft import PeftModel

        peft_loader = PeftModel.from_pretrained

    model_entry = get_model(base_model_id)
    tokenizer = tokenizer_loader(model_entry.repo_id)

    # Must match train.py's build_bnb_config exactly - the adapter was
    # trained against a 4-bit-quantized base model's structure. Loading the
    # base model in full precision here (the original bug) is a real
    # train/inference mismatch, not just slower: it was loading the model
    # without going through the same module wrapping training assumed.
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    base_model = model_loader(model_entry.repo_id, quantization_config=bnb_config, device_map="auto")
    model = peft_loader(base_model, str(adapter_path))

    # Diagnostic, not a fix by itself: if the wrapping mismatch persists
    # after the quantization fix, a peft version installed now that differs
    # from what trained this adapter is the next thing to check - a known
    # cause of exactly this class of key-path error.
    try:
        from peft import PeftConfig
        import peft as _peft_module

        adapter_peft_version = PeftConfig.from_pretrained(str(adapter_path)).peft_version
        if adapter_peft_version and adapter_peft_version != _peft_module.__version__:
            print(
                f"NOTE: this adapter was saved with peft=={adapter_peft_version}, "
                f"but peft=={_peft_module.__version__} is installed now. If loading "
                f"still fails after this fix, a version mismatch is the next thing to check."
            )
    except Exception:
        pass  # diagnostic only - never block loading over a failed version check

    return LoadedModel(model=model, tokenizer=tokenizer, adapter_path=str(adapter_path))


def generate_response(
    loaded: LoadedModel,
    user_message: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate one response from the loaded fine-tuned model, stripping the echoed prompt."""
    prompt = format_chat_prompt(user_message, loaded.tokenizer)
    inputs = loaded.tokenizer(prompt, return_tensors="pt")
    # Move inputs onto whatever device the model actually lives on. Without
    # this, a Colab run surfaced "input_ids is on cpu, whereas the model is
    # on cuda" as a warning rather than a crash - which was luck (VRAM
    # pressure had partially offloaded the model to CPU too, coincidentally
    # narrowing the mismatch), not correctness. Explicit placement here
    # removes that fragility regardless of how the model happens to be sharded.
    inputs = {k: v.to(loaded.model.device) for k, v in inputs.items()}
    output_ids = loaded.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    full_output = loaded.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Slice off exactly the input token length BEFORE decoding, rather than
    # decode-then-string-match. String matching after decode is fundamentally
    # unreliable: apply_chat_template() embeds special tokens (<|user|>,
    # <|end|>, etc.) as literal characters, but decode(skip_special_tokens=
    # True) strips them back out - so decoded text can never literally start
    # with the original templated prompt string. Confirmed this was silently
    # returning the whole echoed prompt instead of just the new generation.
    # Token-level slicing has no such ambiguity.
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    return loaded.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
