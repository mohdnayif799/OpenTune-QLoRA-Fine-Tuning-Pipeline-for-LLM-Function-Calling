"""
tests/test_inference.py

REAL test: format_chat_prompt against a fake tokenizer (same pattern as
data_pipeline's tests) - proves the message structure passed to the chat
template is correct.

MOCKED tests: load_finetuned_model (call order), generate_response (prompt
stripping logic). Prove orchestration is correct - not that a real model
would generate anything coherent. Real verification needs Colab.
"""

from pathlib import Path

from inference import (
    LoadedModel,
    format_chat_prompt,
    generate_response,
    load_finetuned_model,
)


class _FakeTensor:
    """Minimal stand-in for a real torch.Tensor - enough for .to()/.shape/indexing."""

    def __init__(self, value):
        self.value = value
        self.shape = (len(value), len(value[0]))

    def to(self, device):
        return self

    def __getitem__(self, idx):
        return self.value[idx]


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        marker = "[GEN]" if add_generation_prompt else ""
        return " ".join(f"<{m['role']}>{m['content']}" for m in messages) + marker

    def __call__(self, text, return_tensors=None):
        # Pretend the prompt tokenizes to exactly 3 tokens: ids 10/11/12.
        return {"input_ids": _FakeTensor([[10, 11, 12]])}

    def decode(self, ids, skip_special_tokens=True):
        ids = list(ids)
        if ids == [20, 21, 22]:
            # Correct behavior: only the genuinely NEW tokens were sliced off
            # and decoded - this is what the fix should produce.
            return "It is a fine-tuning method."
        # Any other input (e.g. the full sequence, which is what the OLD
        # decode-then-string-match code would have passed) deliberately
        # looks like the bug's real symptom, so a regression is loud, not silent.
        return "<user>What is LoRA?[GEN] It is a fine-tuning method."


# ---------- REAL test ----------

def test_format_chat_prompt_structure():
    prompt = format_chat_prompt("What is LoRA?", _FakeTokenizer())
    assert prompt == "<user>What is LoRA?[GEN]"
    print("PASS: format_chat_prompt produces correctly-structured single-turn prompt")


# ---------- MOCKED tests ----------

def test_load_finetuned_model_calls_in_correct_order():
    calls = []

    def fake_tok_loader(repo_id):
        calls.append(f"tokenizer:{repo_id}")
        return _FakeTokenizer()

    def fake_model_loader(repo_id, quantization_config, device_map):
        calls.append(f"model:{repo_id}")
        return "fake-base-model"

    def fake_peft_loader(base_model, adapter_path):
        calls.append(f"adapter:{adapter_path}")
        return "fake-peft-model"

    loaded = load_finetuned_model(
        base_model_id="mistral-7b",
        adapter_path=Path("/fake/adapter/path"),
        model_loader=fake_model_loader,
        tokenizer_loader=fake_tok_loader,
        peft_loader=fake_peft_loader,
    )

    assert calls[0] == "tokenizer:mistralai/Mistral-7B-Instruct-v0.3"
    assert calls[1] == "model:mistralai/Mistral-7B-Instruct-v0.3"
    # Compare against str(Path(...)) rather than a hardcoded forward-slash
    # string - Path renders with the OS-native separator, so a literal
    # "/fake/adapter/path" only matches by coincidence on Linux and fails on
    # Windows (backslashes). This was a real bug in the test, not in
    # inference.py's actual call order.
    assert calls[2] == f"adapter:{Path('/fake/adapter/path')}"
    assert loaded.model == "fake-peft-model"
    print("PASS: load_finetuned_model loads tokenizer -> base model -> adapter, in that order")


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids, max_new_tokens, temperature, do_sample):
        # Full sequence: 3 prompt tokens (10,11,12) + 3 genuinely new tokens (20,21,22).
        return [[10, 11, 12, 20, 21, 22]]


def test_generate_response_slices_new_tokens_not_string_match():
    """
    Proves the actual fix, not just that some string comes out right:
    generate_response must decode ONLY the new tokens (sliced by input
    length) rather than decode the full sequence and string-match the
    prompt afterward. String matching was the real bug - apply_chat_template
    embeds special tokens as literal text, but decode(skip_special_tokens=
    True) strips them, so a decoded full sequence can never literally start
    with the original prompt string. Confirmed via direct reproduction
    before this fix (see conversation) - real GPU runs were returning the
    entire echoed prompt instead of just the new generation.
    """
    loaded = LoadedModel(model=_FakeModel(), tokenizer=_FakeTokenizer(), adapter_path="/fake")
    response = generate_response(loaded, "What is LoRA?")
    assert response == "It is a fine-tuning method."
    assert "What is LoRA?" not in response
    assert "<user>" not in response  # would appear if the old buggy fallback path fired
    print("PASS: generate_response slices new tokens correctly, not fragile string-matching")


if __name__ == "__main__":
    test_format_chat_prompt_structure()
    test_load_finetuned_model_calls_in_correct_order()
    test_generate_response_slices_new_tokens_not_string_match()
    print("\nAll inference.py tests passed.")
