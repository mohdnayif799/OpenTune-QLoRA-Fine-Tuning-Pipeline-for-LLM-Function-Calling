"""
tests/test_data_pipeline.py

Verifies data_pipeline.py's logic using a FakeTokenizer standing in for a
real HF tokenizer - see module docstring in data_pipeline.py for why.

Covers:
1. A well-formed JSONL file loads correctly into Example objects.
2. A file missing a required column fails with a clear, specific error.
3. An unsupported file extension is rejected.
4. Chat-template formatting passes the right message structure through.
5. Train/val split respects the configured ratio and is deterministic.
6. The full build_dataset pipeline runs end to end against the fake tokenizer.
"""

import json
from pathlib import Path

from config import DatasetConfig
from data_pipeline import (
    build_dataset,
    format_example,
    load_raw_examples,
    split_train_val,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


class FakeTokenizer:
    """
    Minimal stand-in satisfying TokenizerProtocol. Doesn't do real tokenization -
    just enough deterministic behavior to verify the pipeline calls it correctly.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Real templates vary by model; this fake just needs to be inspectable.
        return " ".join(f"<{m['role']}>{m['content']}" for m in messages)

    def __call__(self, text, truncation=True, max_length=128, padding="max_length"):
        # Fake "tokenization": one token per word, padded/truncated to max_length.
        token_ids = [hash(w) % 1000 for w in text.split()][:max_length]
        pad_len = max_length - len(token_ids)
        return {
            "input_ids": token_ids + [0] * pad_len,
            "attention_mask": [1] * len(token_ids) + [0] * pad_len,
        }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_load_valid_jsonl():
    path = FIXTURES_DIR / "valid.jsonl"
    _write_jsonl(path, [{"instruction": "hi", "output": "hello"}] * 3)
    cfg = DatasetConfig(file_path=path)
    examples = load_raw_examples(cfg)
    assert len(examples) == 3
    assert examples[0]["prompt"] == "hi"
    assert examples[0]["response"] == "hello"
    print("PASS: valid JSONL loads into correct Example objects")


def test_missing_column_raises_clear_error():
    path = FIXTURES_DIR / "missing_col.jsonl"
    _write_jsonl(path, [{"instruction": "hi", "wrong_column_name": "hello"}])
    cfg = DatasetConfig(file_path=path)
    try:
        load_raw_examples(cfg)
        raise AssertionError("expected ValueError, none raised")
    except ValueError as e:
        assert "output" in str(e)  # the missing column is named in the error
        print("PASS: missing required column produces a specific, actionable error")


def test_unsupported_extension_rejected():
    path = FIXTURES_DIR / "data.txt"
    path.write_text("not a real dataset")
    cfg = DatasetConfig(file_path=path)
    try:
        load_raw_examples(cfg)
        raise AssertionError("expected ValueError, none raised")
    except ValueError as e:
        assert ".txt" in str(e)
        print("PASS: unsupported file extension is rejected with a clear message")


def test_chat_template_receives_correct_message_structure():
    example = {"prompt": "What is LoRA?", "response": "A low-rank adaptation method."}
    formatted = format_example(example, FakeTokenizer())
    assert "<user>What is LoRA?" in formatted
    assert "<assistant>A low-rank adaptation method." in formatted
    print("PASS: chat template receives correctly-structured user/assistant messages")


def test_split_is_proportional_and_deterministic():
    examples = [{"prompt": str(i), "response": str(i)} for i in range(20)]
    train_a, val_a = split_train_val(examples, validation_split=0.2, seed=42)
    train_b, val_b = split_train_val(examples, validation_split=0.2, seed=42)

    assert len(val_a) == 4  # 20% of 20
    assert len(train_a) == 16
    assert train_a == train_b and val_a == val_b  # same seed -> same split
    print(f"PASS: split is proportional (16 train / 4 val) and deterministic across runs")


def test_build_dataset_end_to_end():
    path = FIXTURES_DIR / "full_pipeline.jsonl"
    _write_jsonl(
        path,
        [{"instruction": f"question {i}", "output": f"answer {i}"} for i in range(10)],
    )
    cfg = DatasetConfig(file_path=path, validation_split=0.2, max_sequence_length=32)
    train, val = build_dataset(cfg, FakeTokenizer())

    assert len(train) == 8 and len(val) == 2
    assert len(train[0]["input_ids"]) == 32  # padded to max_sequence_length
    assert len(train[0]["attention_mask"]) == 32
    assert "question" in train[0]["formatted_text"]
    print("PASS: full build_dataset pipeline runs end to end (load -> split -> format -> tokenize)")


if __name__ == "__main__":
    test_load_valid_jsonl()
    test_missing_column_raises_clear_error()
    test_unsupported_extension_rejected()
    test_chat_template_receives_correct_message_structure()
    test_split_is_proportional_and_deterministic()
    test_build_dataset_end_to_end()
    print("\nAll data_pipeline.py tests passed.")
