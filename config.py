"""
config.py

Typed configuration schemas for OpenTune fine-tuning runs.

Per the design document's reproducibility requirement (Section 7), a run's
config is the single source of truth logged alongside its results — so any
run can be inspected or reproduced later from this object alone. Every other
module (model_registry, data_pipeline, train, evaluate) consumes these
schemas rather than passing around loose dicts or keyword arguments.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from model_registry import MODEL_REGISTRY


class QuantizationType(str, Enum):
    """Supported quantization modes for loading the base model in 4-bit."""

    NF4 = "nf4"
    FP4 = "fp4"


class LoRAPreset(str, Enum):
    """Named LoRA presets, per design doc Section 17."""

    FAST_LOW_VRAM = "fast_low_vram"
    BALANCED = "balanced"
    HIGHER_QUALITY = "higher_quality"
    CUSTOM = "custom"


# Preset -> concrete (rank, alpha) values, matching design doc Section 17's table.
# CUSTOM is deliberately absent: it means "use whatever the user explicitly passed."
LORA_PRESET_VALUES: dict[LoRAPreset, dict[str, int]] = {
    LoRAPreset.FAST_LOW_VRAM: {"r": 8, "lora_alpha": 16},
    LoRAPreset.BALANCED: {"r": 16, "lora_alpha": 32},
    LoRAPreset.HIGHER_QUALITY: {"r": 32, "lora_alpha": 64},
}


class LoRAConfig(BaseModel):
    """
    LoRA/QLoRA hyperparameters for a single fine-tuning run.

    If `preset` is anything other than CUSTOM, `r` and `lora_alpha` are
    overwritten from LORA_PRESET_VALUES after validation — the preset is the
    source of truth in that case, not whatever was passed in for those two
    fields. Pass preset=CUSTOM to take direct control of r/lora_alpha.
    """

    preset: LoRAPreset = LoRAPreset.BALANCED
    r: int = Field(default=16, gt=0, description="LoRA rank")
    lora_alpha: int = Field(default=32, gt=0, description="LoRA scaling factor")
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Attention projection layers to adapt; model-family specific "
        "(see model_registry.py, which supplies the correct value per model).",
    )
    quantization: QuantizationType = QuantizationType.NF4
    gradient_checkpointing: bool = True

    def model_post_init(self, __context) -> None:
        """Apply preset values after field validation, unless preset is CUSTOM."""
        if self.preset != LoRAPreset.CUSTOM:
            preset_values = LORA_PRESET_VALUES[self.preset]
            self.r = preset_values["r"]
            self.lora_alpha = preset_values["lora_alpha"]


class DatasetConfig(BaseModel):
    """Dataset source location and column-mapping configuration."""

    file_path: Path
    prompt_column: str = "instruction"
    response_column: str = "output"
    validation_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    max_sequence_length: int = Field(default=1024, gt=0)

    @field_validator("file_path")
    @classmethod
    def file_must_exist(cls, v: Path) -> Path:
        """Fail fast at config-creation time, not hours into a training run."""
        if not v.exists():
            raise ValueError(f"Dataset file not found: {v}")
        return v


class TrainingConfig(BaseModel):
    """Training loop hyperparameters."""

    num_train_epochs: int = Field(default=3, gt=0)
    per_device_train_batch_size: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    checkpoint_every_n_steps: int = Field(default=50, gt=0)
    early_stopping_patience: Optional[int] = Field(
        default=3,
        description="Stop if validation loss doesn't improve for N eval "
        "checks in a row. None disables early stopping.",
    )
    seed: int = 42
    force_cpu: bool = Field(
        default=False,
        description="Explicitly train on CPU instead of GPU. Per design doc "
        "Section 25, execution environment is selected via this config flag, "
        "not runtime auto-detection - SFTConfig validates bf16/GPU support at "
        "construction time and fails ungracefully without this being explicit.",
    )


class RunConfig(BaseModel):
    """
    Top-level configuration for one complete OpenTune fine-tuning run.

    This is the object serialized to logs/<run_id>/config.json at the start
    of a run (design doc Section 22), making every run fully reproducible
    from that one file.
    """

    run_name: str
    base_model_id: str = Field(
        description="A key from model_registry.MODEL_REGISTRY, e.g. 'mistral-7b'. "
        "NOT a raw Hugging Face repo ID - the registry entry supplies that, "
        "along with the architecture-correct LoRA target_modules."
    )
    dataset: DatasetConfig
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @field_validator("base_model_id")
    @classmethod
    def model_must_be_registered(cls, v: str) -> str:
        if v not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"'{v}' is not a registered model. Available: {available}")
        return v
