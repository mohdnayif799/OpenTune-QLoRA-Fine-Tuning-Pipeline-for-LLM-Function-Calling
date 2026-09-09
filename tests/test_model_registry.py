"""
tests/test_model_registry.py

Verifies:
1. Every registered model resolves correctly by key.
2. An unknown key raises a clear, actionable error (not a bare KeyError).
3. list_models(include_gated=False) actually filters gated models out.
4. Phi-3's target_modules are the fused-projection names, not the Llama-style
   list - this is the specific bug the module docstring warns about, so it
   gets its own explicit test rather than being assumed.
"""

from model_registry import MODEL_REGISTRY, get_model, list_models


def test_all_registered_models_resolve():
    for key in MODEL_REGISTRY:
        entry = get_model(key)
        assert entry.repo_id, f"{key} has an empty repo_id"
    print(f"PASS: all {len(MODEL_REGISTRY)} registered models resolve correctly")


def test_unknown_key_raises_clear_error():
    try:
        get_model("not-a-real-model")
        raise AssertionError("expected KeyError, none raised")
    except KeyError as e:
        assert "Unknown model key" in str(e)
        assert "mistral-7b" in str(e)  # confirms available models are listed
        print("PASS: unknown key raises a clear, actionable error")


def test_gated_filter_excludes_gated_models():
    all_models = list_models(include_gated=True)
    ungated_only = list_models(include_gated=False)
    assert len(ungated_only) < len(all_models)
    assert all(not m.is_gated for m in ungated_only)
    print(f"PASS: gated filter correctly excludes gated models "
          f"({len(all_models)} total -> {len(ungated_only)} ungated)")


def test_phi3_uses_fused_target_modules_not_llama_style():
    phi3 = get_model("phi-3-mini")
    assert "qkv_proj" in phi3.target_modules
    assert "gate_up_proj" in phi3.target_modules
    # The Llama-style separate names should NOT be present for Phi-3.
    assert "q_proj" not in phi3.target_modules
    assert "gate_proj" not in phi3.target_modules
    print("PASS: Phi-3 uses correct fused target_modules (qkv_proj/gate_up_proj)")


if __name__ == "__main__":
    test_all_registered_models_resolve()
    test_unknown_key_raises_clear_error()
    test_gated_filter_excludes_gated_models()
    test_phi3_uses_fused_target_modules_not_llama_style()
    print("\nAll model_registry.py tests passed.")
