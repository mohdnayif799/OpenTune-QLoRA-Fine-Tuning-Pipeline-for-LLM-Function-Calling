"""
tests/test_app.py

Uses Streamlit's real AppTest framework (headless, no browser) to verify
the app actually renders and is wired correctly.

IMPORTANT boundary: these tests do NOT click "Start Fine-Tuning", "Run
Evaluation", or send a chat message - doing so would trigger real calls into
train.run_training / evaluate_models / generate_response, which need a GPU
and real model weights neither this sandbox nor these tests have. What's
tested here is that the app renders without error and every expected widget
exists with correct options - not that the training/eval/chat flows work
end to end. That needs your Colab environment, same boundary as train.py.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest.from_file() first checks whether its argument resolves as a file
# relative to the CURRENT WORKING DIRECTORY - and only if that fails, falls
# back to walking the call stack to guess the caller's location (Streamlit's
# own source marks this fallback "TODO: Make this not super fragile"). Rather
# than depend on either behavior, resolve an absolute path ourselves so this
# works identically no matter what directory these tests are invoked from.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _new_app_test() -> AppTest:
    """
    Every test gets a fresh AppTest instance from the same absolute path.

    default_timeout is set well above AppTest's 3-second default: app.py
    imports transformers/peft/trl/rouge_score at module level, and a cold
    import of that stack measured well over a minute in testing - nowhere
    close to fitting in 3 seconds. This is not a flaky retry fix; it's
    sizing the timeout to match a real, measured cost.
    """
    return AppTest.from_file(APP_PATH, default_timeout=120)


def test_app_runs_without_error():
    at = _new_app_test()
    at.run()
    assert not at.exception, f"App raised on initial render: {at.exception}"
    print("PASS: app.py renders without raising - the models_by_key ordering bug is fixed")


def test_model_dropdown_lists_all_registered_models():
    at = _new_app_test()
    at.run()
    selectboxes = at.selectbox
    model_dropdown = selectboxes[0]  # first selectbox in the app is the model picker
    assert len(model_dropdown.options) == 5  # matches MODEL_REGISTRY size
    assert any("Phi-3" in opt for opt in model_dropdown.options)
    assert any("gated" in opt.lower() for opt in model_dropdown.options)
    print(f"PASS: model dropdown lists all {len(model_dropdown.options)} registered models, "
          f"including gated ones")


def test_gated_model_selection_shows_warning():
    at = _new_app_test()
    at.run()
    at.selectbox[0].select("Llama 3 8B Instruct (gated)").run()
    warnings = [w.value for w in at.warning]
    assert any("gated" in w.lower() for w in warnings)
    print("PASS: selecting a gated model surfaces the license warning")


def test_lora_preset_radio_has_all_four_options():
    at = _new_app_test()
    at.run()
    preset_radio = at.radio[0]
    assert set(preset_radio.options) == {"fast_low_vram", "balanced", "higher_quality", "custom"}
    print("PASS: LoRA preset radio exposes all four presets from config.py")


def test_all_four_tabs_render():
    at = _new_app_test()
    at.run()
    assert not at.exception
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == ["1. Setup", "2. Configure & Train", "3. Evaluate", "4. Chat"]
    print("PASS: all four design-doc views (Setup/Train/Evaluate/Chat) render as tabs")


def test_evaluate_tab_shows_prompt_when_no_training_yet():
    at = _new_app_test()
    at.run()
    info_messages = [i.value for i in at.info]
    assert any("training run" in msg.lower() for msg in info_messages)
    print("PASS: Evaluate tab correctly prompts for a completed training run first")


def test_chat_tab_locked_with_nothing_loaded_yet():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    info_messages = [i.value for i in at.info]
    assert any("load an already-trained adapter" in msg.lower() for msg in info_messages), (
        "Chat tab's locked message should mention the load-adapter path now, not just training"
    )
    print("PASS: Chat tab's gate message reflects both unlock paths (train or load)")


def test_setup_tab_has_load_adapter_button():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    button_labels = [b.label for b in at.button]
    assert "Load Adapter" in button_labels, "Setup tab must expose the load-existing-adapter button"
    print("PASS: 'Load Adapter' button is present in Setup tab")


if __name__ == "__main__":
    test_app_runs_without_error()
    test_model_dropdown_lists_all_registered_models()
    test_gated_model_selection_shows_warning()
    test_lora_preset_radio_has_all_four_options()
    test_all_four_tabs_render()
    test_evaluate_tab_shows_prompt_when_no_training_yet()
    test_chat_tab_locked_with_nothing_loaded_yet()
    test_setup_tab_has_load_adapter_button()
    print("\nAll app.py structural tests passed (UI wiring - NOT the GPU-bound flows).")
