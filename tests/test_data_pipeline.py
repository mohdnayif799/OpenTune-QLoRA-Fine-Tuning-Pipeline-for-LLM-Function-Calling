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
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

from config import DatasetConfig
from data_pipeline import (
    build_dataset,
    format_example,
    held_out_examples,
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


def _write_bytes_to_temp(name: str, data: bytes) -> Path:
    """Edge-case files go to a fresh temp folder so the tracked fixtures stay untouched."""
    path = Path(tempfile.mkdtemp()) / name
    path.write_bytes(data)
    return path


def test_utf8_text_is_read_as_utf8_on_every_platform():
    rows = [{"instruction": "Qu'est-ce qu'un café? 日本語", "output": "Réponse naïve €"}]
    jsonl = _write_bytes_to_temp("nonascii.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode("utf-8"))
    csv_path = _write_bytes_to_temp("nonascii.csv", "instruction,output\ncafé,naïve €\n".encode("utf-8"))
    assert load_raw_examples(DatasetConfig(file_path=jsonl))[0] == {"prompt": rows[0]["instruction"], "response": rows[0]["output"]}
    assert load_raw_examples(DatasetConfig(file_path=csv_path))[0] == {"prompt": "café", "response": "naïve €"}
    print("PASS: UTF-8 datasets decode as UTF-8 regardless of the platform's default code page")


def test_leading_bom_does_not_corrupt_the_first_column():
    bom = b"\xef\xbb\xbf"
    csv_path = _write_bytes_to_temp("bom.csv", bom + b"instruction,output\nq1,a1\n")
    jsonl = _write_bytes_to_temp("bom.jsonl", bom + b'{"instruction": "q1", "output": "a1"}\n')
    assert load_raw_examples(DatasetConfig(file_path=csv_path)) == [{"prompt": "q1", "response": "a1"}]
    assert load_raw_examples(DatasetConfig(file_path=jsonl)) == [{"prompt": "q1", "response": "a1"}]
    print("PASS: an Excel style UTF-8 BOM is dropped instead of becoming part of the first header")


def test_non_utf8_file_falls_back_to_the_platform_default_with_a_warning():
    # cp1252 bytes for "café" are not valid UTF-8. Before the UTF-8 change such
    # a file loaded on a cp1252 machine, so it must still load there.
    csv_path = _write_bytes_to_temp("cp1252.csv", "instruction,output\ncafé,ok\n".encode("cp1252"))
    with patch("locale.getpreferredencoding", return_value="cp1252"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            examples = load_raw_examples(DatasetConfig(file_path=csv_path))
    assert examples == [{"prompt": "café", "response": "ok"}]
    assert any("UTF-8" in str(w.message) for w in caught)
    print("PASS: a non UTF-8 file still loads through the old platform default, with a warning")


def test_uppercase_suffixes_are_accepted():
    jsonl = _write_bytes_to_temp("UPPER.JSONL", b'{"instruction": "q", "output": "a"}\n')
    csv_path = _write_bytes_to_temp("UPPER.CSV", b"instruction,output\nq,a\n")
    assert load_raw_examples(DatasetConfig(file_path=jsonl)) == [{"prompt": "q", "response": "a"}]
    assert load_raw_examples(DatasetConfig(file_path=csv_path)) == [{"prompt": "q", "response": "a"}]
    print("PASS: .JSONL and .CSV are read the same as .jsonl and .csv")


def test_later_row_missing_a_column_names_the_row_and_column():
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(5)] + [{"instruction": "q5"}]
    path = _write_bytes_to_temp("missing_later.jsonl", "\n".join(json.dumps(r) for r in rows).encode("utf-8"))
    try:
        load_raw_examples(DatasetConfig(file_path=path))
        raise AssertionError("expected ValueError, none raised")
    except ValueError as e:
        assert "Row 6" in str(e) and "'output'" in str(e), str(e)
    print("PASS: a later row without a required column raises ValueError naming row 6 and 'output'")


def test_jsonl_line_that_is_not_an_object_raises_a_clear_error():
    path = _write_bytes_to_temp("array.jsonl", json.dumps([{"instruction": "q", "output": "a"}]).encode("utf-8"))
    try:
        load_raw_examples(DatasetConfig(file_path=path))
        raise AssertionError("expected ValueError, none raised")
    except ValueError as e:
        assert "Row 1" in str(e) and "object" in str(e), str(e)
    print("PASS: a JSON array saved as .jsonl raises a ValueError instead of an AttributeError")


def _messages_of(caught) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def test_non_string_and_empty_values_warn_but_still_load():
    rows = [
        {"instruction": "q1", "output": None},
        {"instruction": "q2", "output": 42},
        {"instruction": "", "output": "a3"},
        {"instruction": "q4", "output": "a4"},
    ]
    path = _write_bytes_to_temp("odd_values.jsonl", "\n".join(json.dumps(r) for r in rows).encode("utf-8"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        examples = load_raw_examples(DatasetConfig(file_path=path))
    assert len(examples) == 4 and examples[1]["response"] == 42  # values pass through unchanged
    messages = _messages_of(caught)
    assert any("not text" in m and "[1, 2]" in m for m in messages), messages
    assert any("empty" in m and "[3]" in m for m in messages), messages
    print("PASS: non-string and empty values load unchanged and produce warnings naming their rows")


def test_examples_that_fill_max_sequence_length_warn():
    path = _write_bytes_to_temp(
        "long.jsonl",
        "\n".join(json.dumps({"instruction": "q", "output": "word " * 40}) for _ in range(4)).encode("utf-8"),
    )
    cfg = DatasetConfig(file_path=path, validation_split=0.25, max_sequence_length=8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train, val = build_dataset(cfg, FakeTokenizer())
    assert len(train) == 3 and len(val) == 1  # still built, only warned
    assert any("max_sequence_length=8" in m for m in _messages_of(caught)), _messages_of(caught)
    print("PASS: examples cut at max_sequence_length are reported with a warning, not an error")


def test_clean_dataset_produces_no_warnings():
    path = _write_bytes_to_temp(
        "clean.jsonl",
        "\n".join(json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) for i in range(5)).encode("utf-8"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_dataset(DatasetConfig(file_path=path, max_sequence_length=32), FakeTokenizer())
    assert _messages_of(caught) == []
    print("PASS: a clean dataset builds without any warnings")


def test_held_out_examples_come_from_the_validation_slice_only():
    path = _write_bytes_to_temp(
        "split.jsonl",
        "\n".join(json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) for i in range(20)).encode("utf-8"),
    )
    cfg = DatasetConfig(file_path=path, validation_split=0.2)
    held_out = held_out_examples(cfg, 3)

    train_part, val_part = split_train_val(load_raw_examples(cfg), cfg.validation_split)
    assert len(held_out) == 3
    assert held_out == val_part[:3]
    assert not any(ex in train_part for ex in held_out)
    # Same slice build_dataset holds out from training, not just the same function.
    _, val_tokenized = build_dataset(cfg, FakeTokenizer())
    assert [format_example(ex, FakeTokenizer()) for ex in held_out] == [
        v["formatted_text"] for v in val_tokenized[:3]
    ]
    print("PASS: held_out_examples returns the first examples of build_dataset's validation slice")


if __name__ == "__main__":
    test_load_valid_jsonl()
    test_missing_column_raises_clear_error()
    test_unsupported_extension_rejected()
    test_chat_template_receives_correct_message_structure()
    test_split_is_proportional_and_deterministic()
    test_build_dataset_end_to_end()
    test_utf8_text_is_read_as_utf8_on_every_platform()
    test_leading_bom_does_not_corrupt_the_first_column()
    test_non_utf8_file_falls_back_to_the_platform_default_with_a_warning()
    test_uppercase_suffixes_are_accepted()
    test_later_row_missing_a_column_names_the_row_and_column()
    test_jsonl_line_that_is_not_an_object_raises_a_clear_error()
    test_non_string_and_empty_values_warn_but_still_load()
    test_examples_that_fill_max_sequence_length_warn()
    test_clean_dataset_produces_no_warnings()
    test_held_out_examples_come_from_the_validation_slice_only()
    print("\nAll data_pipeline.py tests passed.")
