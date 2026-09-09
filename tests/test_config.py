"""
tests/test_config.py

Verifies config.py's core guarantees:
1. Preset resolution actually overwrites r/alpha with the preset's values.
2. CUSTOM preset preserves explicitly-passed r/alpha instead of overwriting them.
3. A nonexistent dataset file is rejected at config-creation time, not later.
4. An out-of-range validation_split is rejected.
5. A RunConfig round-trips to JSON and back — required for the reproducibility
   guarantee in design doc Section 7/22 (a run must be reconstructable from
   its logged config alone).
"""

from pathlib import Path
from pydantic import ValidationError

from config import DatasetConfig, LoRAConfig, LoRAPreset, RunConfig

# A real file has to exist for DatasetConfig's validator to accept it.
DUMMY_DATASET = Path(__file__).parent / "dummy_dataset.jsonl"
DUMMY_DATASET.write_text('{"instruction": "test", "output": "test"}\n')


def test_preset_overwrites_r_and_alpha():
    # Deliberately pass WRONG r/alpha alongside a BALANCED preset, to prove
    # the preset wins rather than silently accepting the wrong values.
    cfg = LoRAConfig(preset=LoRAPreset.BALANCED, r=999, lora_alpha=999)
    assert cfg.r == 16, f"expected preset to set r=16, got {cfg.r}"
    assert cfg.lora_alpha == 32, f"expected preset to set lora_alpha=32, got {cfg.lora_alpha}"
    print("PASS: BALANCED preset correctly overwrites r/alpha to (16, 32)")


def test_custom_preset_preserves_explicit_values():
    cfg = LoRAConfig(preset=LoRAPreset.CUSTOM, r=64, lora_alpha=128)
    assert cfg.r == 64, f"expected CUSTOM to preserve r=64, got {cfg.r}"
    assert cfg.lora_alpha == 128, f"expected CUSTOM to preserve lora_alpha=128, got {cfg.lora_alpha}"
    print("PASS: CUSTOM preset preserves explicitly-passed r/alpha")


def test_nonexistent_dataset_file_rejected():
    try:
        DatasetConfig(file_path=Path("this_file_does_not_exist.jsonl"))
        raise AssertionError("expected ValidationError, but no error was raised")
    except ValidationError as e:
        assert "not found" in str(e)
        print("PASS: nonexistent dataset file is correctly rejected at config time")


def test_invalid_validation_split_rejected():
    try:
        DatasetConfig(file_path=DUMMY_DATASET, validation_split=1.5)
        raise AssertionError("expected ValidationError, but no error was raised")
    except ValidationError:
        print("PASS: out-of-range validation_split (1.5) is correctly rejected")


def test_runconfig_rejects_unregistered_model():
    try:
        RunConfig(
            run_name="test",
            base_model_id="not-a-registered-model",
            dataset=DatasetConfig(file_path=DUMMY_DATASET),
        )
        raise AssertionError("expected ValidationError, none raised")
    except ValidationError as e:
        assert "not a registered model" in str(e)
        print("PASS: RunConfig rejects a base_model_id that isn't in model_registry")


def test_runconfig_roundtrips_through_json():
    original = RunConfig(
        run_name="test-run-001",
        base_model_id="mistral-7b",
        dataset=DatasetConfig(file_path=DUMMY_DATASET),
    )
    json_str = original.model_dump_json()
    restored = RunConfig.model_validate_json(json_str)

    assert restored.run_name == original.run_name
    assert restored.base_model_id == original.base_model_id
    assert restored.lora.r == original.lora.r == 16  # BALANCED default
    assert restored.dataset.file_path == original.dataset.file_path
    print("PASS: RunConfig survives a full JSON round-trip (reproducibility requirement)")


if __name__ == "__main__":
    test_preset_overwrites_r_and_alpha()
    test_custom_preset_preserves_explicit_values()
    test_nonexistent_dataset_file_rejected()
    test_invalid_validation_split_rejected()
    test_runconfig_rejects_unregistered_model()
    test_runconfig_roundtrips_through_json()
    DUMMY_DATASET.unlink()
    print("\nAll config.py tests passed.")
