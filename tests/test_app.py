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

The Chat tab tests at the end are the one exception: they unlock the tab with
a stand-in object instead of a loaded model and patch
inference.generate_response, so the rendering and the submit flow run
without a GPU. They still say nothing about what a real model would answer.
"""

from pathlib import Path
from unittest.mock import patch

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


def _new_unlocked_chat_app_test(history: list[tuple[str, str]]) -> AppTest:
    """
    The Chat tab only checks that loaded_chat_model is not None, so a plain
    object() unlocks it without loading weights. Seeding chat_history before
    the first run is how a returning user's conversation looks to the script.
    """
    at = _new_app_test()
    at.session_state["loaded_chat_model"] = object()
    at.session_state["chat_history"] = list(history)
    return at


def _user_bubbles(at: AppTest) -> list[str]:
    """The CSS block mentions .user-msg too, so match the class attribute, not the name."""
    return [m.value for m in at.markdown if 'class="user-msg"' in m.value]


def test_chat_tab_renders_without_chat_message_avatars():
    at = _new_unlocked_chat_app_test([("user", "hello"), ("assistant", "hi there")])
    at.run()
    assert not at.exception, f"Chat tab raised on render: {at.exception}"
    assert len(at.chat_message) == 0
    print("PASS: Chat tab renders history without st.chat_message, so no avatars are drawn")


def test_chat_user_bubble_escapes_html_and_keeps_markdown_literal():
    at = _new_unlocked_chat_app_test([("user", "<b>x</b> *y* __init__.py\n\nz")])
    at.run()
    assert not at.exception
    bubbles = _user_bubbles(at)
    assert len(bubbles) == 1
    bubble = bubbles[0]
    assert "&lt;b&gt;x&lt;/b&gt;" in bubble
    assert "*y* __init__.py" in bubble
    assert "<b>" not in bubble
    assert "\n" not in bubble
    print("PASS: user bubble shows HTML and Markdown characters exactly as typed, on one HTML line")


def test_chat_assistant_turn_renders_as_plain_markdown():
    answer = "Plain answer with **bold** text."
    at = _new_unlocked_chat_app_test([("user", "q"), ("assistant", answer)])
    at.run()
    assert not at.exception
    assert answer in [m.value for m in at.markdown]
    print("PASS: assistant turn is a bare markdown element with its exact text")


def test_chat_submit_appends_one_turn_each_and_renders_answer():
    at = _new_unlocked_chat_app_test([])
    # app.py does `from inference import generate_response` on every script
    # run, so patching the module attribute is what the script picks up.
    with patch("inference.generate_response", return_value="patched answer"):
        at.run()
        at.chat_input[0].set_value("hi").run()
    assert not at.exception, f"Chat submit raised: {at.exception}"
    assert at.session_state["chat_history"] == [("user", "hi"), ("assistant", "patched answer")]
    assert len(_user_bubbles(at)) == 1
    assert "patched answer" in [m.value for m in at.markdown]
    print("PASS: submitting a question adds one user and one assistant turn and renders both")


def _start_fine_tuning_with_fake_training(upload_name: str, learning_rate: float | None = None):
    """
    Drive the Train tab up to run_training without a GPU: the upload goes
    through the real file_uploader widget, and train.run_training is patched
    (app.py imports it by name on every script run) to record the RunConfig
    it was handed instead of training.
    """
    from train import TrainingResult

    received = []

    def fake_run_training(run_config, checkpoints_dir, logs_dir):
        received.append(run_config)
        return TrainingResult(
            run_name=run_config.run_name, adapter_path="fake", final_train_loss=0.5,
            final_eval_loss=None, num_steps_completed=1, stopped_early=False,
        )

    at = _new_app_test()
    with patch("train.run_training", side_effect=fake_run_training):
        at.run()
        at.file_uploader[0].set_value(
            (upload_name, b'{"instruction": "q", "output": "a"}\n', "application/json")
        ).run()
        if learning_rate is not None:
            [n for n in at.number_input if n.label == "Learning rate"][0].set_value(learning_rate).run()
        [b for b in at.button if b.label == "Start Fine-Tuning"][0].click().run()
    return at, received


def test_uploaded_dataset_is_written_to_the_system_temp_dir_by_base_name():
    import tempfile

    at, received = _start_fine_tuning_with_fake_training("nested/dir/upload_probe.jsonl")
    assert not at.exception, f"Start Fine-Tuning raised: {at.exception}"
    assert len(received) == 1
    written = received[0].dataset.file_path
    try:
        assert written.parent == Path(tempfile.gettempdir())
        assert written.name == "upload_probe.jsonl"
    finally:
        written.unlink(missing_ok=True)
    print("PASS: upload is saved under tempfile.gettempdir() using only the file's base name")


if __name__ == "__main__":
    test_app_runs_without_error()
    test_model_dropdown_lists_all_registered_models()
    test_gated_model_selection_shows_warning()
    test_lora_preset_radio_has_all_four_options()
    test_all_four_tabs_render()
    test_evaluate_tab_shows_prompt_when_no_training_yet()
    test_chat_tab_locked_with_nothing_loaded_yet()
    test_setup_tab_has_load_adapter_button()
    test_chat_tab_renders_without_chat_message_avatars()
    test_chat_user_bubble_escapes_html_and_keeps_markdown_literal()
    test_chat_assistant_turn_renders_as_plain_markdown()
    test_chat_submit_appends_one_turn_each_and_renders_answer()
    test_uploaded_dataset_is_written_to_the_system_temp_dir_by_base_name()
    print("\nAll app.py structural tests passed (UI wiring - NOT the GPU-bound flows).")
