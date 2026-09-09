"""
data_pipeline.py

Dataset loading, schema validation, chat-template formatting, and
tokenization - per design doc Section 16.

Functions here accept a tokenizer typed as TokenizerProtocol rather than
importing transformers.AutoTokenizer directly. This is a deliberate
dependency-injection choice: the pipeline logic (column validation,
splitting, template application) doesn't actually need a *real* tokenizer to
be correct, only something with the right interface - so it can be fully
unit-tested with a lightweight fake (see tests/test_data_pipeline.py)
without a network call or GPU. The real training/inference modules pass in
an actual HF AutoTokenizer, which satisfies the same protocol.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Protocol, TypedDict

from config import DatasetConfig


class TokenizerProtocol(Protocol):
    """The minimal tokenizer interface this module needs - see module docstring."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def __call__(self, text: str, truncation: bool, max_length: int, padding: str) -> dict: ...


class Example(TypedDict):
    prompt: str
    response: str


class TokenizedExample(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    formatted_text: str


def load_raw_examples(config: DatasetConfig) -> list[Example]:
    """
    Load a CSV or JSONL file and map its configured columns to {prompt, response}.

    Raises ValueError showing expected vs. actual columns if the mapping
    doesn't match what's in the file - the schema-mismatch failure mode
    from design doc Section 23, surfaced here rather than as a raw KeyError
    deep inside training.
    """
    path = config.file_path
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".csv":
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported dataset format: '{path.suffix}'. Expected .csv or .jsonl")

    if not rows:
        raise ValueError(f"Dataset file {path} contains no rows")

    actual_columns = set(rows[0].keys())
    required = {config.prompt_column, config.response_column}
    missing = required - actual_columns
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {sorted(missing)}. "
            f"Expected: {sorted(required)}. Found: {sorted(actual_columns)}"
        )

    return [
        Example(prompt=row[config.prompt_column], response=row[config.response_column])
        for row in rows
    ]


def format_example(example: Example, tokenizer: TokenizerProtocol) -> str:
    """Apply the selected model's own chat template (design doc Section 16, point 2)."""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def tokenize_example(
    formatted_text: str, tokenizer: TokenizerProtocol, max_length: int
) -> TokenizedExample:
    """Tokenize already-formatted text with truncation/padding to max_length."""
    encoded = tokenizer(formatted_text, truncation=True, max_length=max_length, padding="max_length")
    return TokenizedExample(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        formatted_text=formatted_text,
    )


def split_train_val(
    examples: list[Example], validation_split: float, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Deterministically (seeded) shuffle and split into train/validation sets."""
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * validation_split)) if len(shuffled) > 1 else 0
    return shuffled[val_size:], shuffled[:val_size]


def build_dataset(
    config: DatasetConfig, tokenizer: TokenizerProtocol
) -> tuple[list[TokenizedExample], list[TokenizedExample]]:
    """Full pipeline: load -> validate -> split -> format -> tokenize."""
    raw_examples = load_raw_examples(config)
    train_raw, val_raw = split_train_val(raw_examples, config.validation_split)

    def process(examples: list[Example]) -> list[TokenizedExample]:
        return [
            tokenize_example(format_example(ex, tokenizer), tokenizer, config.max_sequence_length)
            for ex in examples
        ]

    return process(train_raw), process(val_raw)
