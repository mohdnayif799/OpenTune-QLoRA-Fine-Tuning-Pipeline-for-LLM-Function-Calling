"""
model_registry.py

The supported base-model catalog, per design doc Section 14.

Each entry carries the model-family-specific metadata that config.py and
data_pipeline.py need but shouldn't have to know themselves: the correct
LoRA target module names for that architecture (Section 15), whether the
model is gated, and its approximate size for UI/VRAM-budgeting purposes.

IMPORTANT: target_modules is NOT the same generic list across every model.
Phi-3 in particular fuses its attention and MLP projections into
qkv_proj/gate_up_proj rather than separate q_proj/k_proj/v_proj/gate_proj/
up_proj — using the Llama-style list on Phi-3 fails at training time with
"Target modules {...} not found in the base model." This is a well-known,
easy-to-hit mistake; verified against Microsoft's own PEFT usage examples
before being hardcoded here.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelEntry(BaseModel):
    """Everything the rest of OpenTune needs to know about one base model."""

    repo_id: str
    display_name: str
    param_count_billions: float
    is_gated: bool
    target_modules: list[str]
    recommended_max_seq_length: int = 2048


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "phi-3-mini": ModelEntry(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        display_name="Phi-3 Mini (3.8B)",
        param_count_billions=3.8,
        is_gated=False,
        # Fused projections - see module docstring above.
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        recommended_max_seq_length=4096,
    ),
    "mistral-7b": ModelEntry(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        display_name="Mistral 7B Instruct",
        param_count_billions=7.0,
        is_gated=False,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        recommended_max_seq_length=2048,
    ),
    "qwen2.5-7b": ModelEntry(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen2.5 7B Instruct",
        param_count_billions=7.0,
        is_gated=False,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        recommended_max_seq_length=2048,
    ),
    "gemma-2-9b": ModelEntry(
        repo_id="google/gemma-2-9b-it",
        display_name="Gemma 2 9B (gated)",
        param_count_billions=9.0,
        is_gated=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        recommended_max_seq_length=2048,
    ),
    "llama-3-8b": ModelEntry(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        display_name="Llama 3 8B Instruct (gated)",
        param_count_billions=8.0,
        is_gated=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        recommended_max_seq_length=2048,
    ),
}


def get_model(key: str) -> ModelEntry:
    """Look up a model by its registry key. Raises a clear error if unknown."""
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model key '{key}'. Available: {available}")
    return MODEL_REGISTRY[key]


def list_models(include_gated: bool = True) -> list[ModelEntry]:
    """Return all registry entries, optionally excluding gated models."""
    models = list(MODEL_REGISTRY.values())
    if not include_gated:
        models = [m for m in models if not m.is_gated]
    return models
