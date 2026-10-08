"""
app.py

The Streamlit UI tying every module together, per design doc Section 21:
four views - Setup (model + dataset), Configure & Train, Evaluate, Chat.

This module is UI wiring, not business logic - it should call into config.py/
model_registry.py/data_pipeline.py/train.py/evaluate.py/inference.py rather
than reimplement anything. Real training/evaluation/generation calls only
make sense when this app is actually running on a GPU-backed machine (a
Colab tunnel or a rented GPU box) - see design doc Section 25.
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

import streamlit as st

from config import DatasetConfig, LoRAConfig, LoRAPreset, RunConfig, TrainingConfig
from model_registry import MODEL_REGISTRY, get_model
from data_pipeline import held_out_examples
from evaluate import evaluate_models
from inference import LoadedModel, generate_response, load_finetuned_model
from train import run_training

st.set_page_config(page_title="OpenTune", layout="wide")
st.title("OpenTune")
st.caption("Fine-tune open-weight LLMs with QLoRA, evaluated against the base model.")

# --- Session state defaults ---
for key, default in {
    "run_config": None,
    "training_result": None,
    "comparison_report": None,
    "loaded_chat_model": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

CHAT_CSS = """
<style>
.user-msg-row {
    display: flex;
    justify-content: flex-end;
    margin: 1.75rem 0 0.55rem 0;
}
.user-msg {
    max-width: min(56%, 36rem);
    background: rgba(59, 130, 246, 0.10);
    border: 1px solid rgba(59, 130, 246, 0.20);
    border-radius: 1.15rem 1.15rem 0.35rem 1.15rem;
    padding: 0.7rem 1.05rem;
    font-size: 0.95rem;
    line-height: 1.55;
    text-align: left;
    overflow-wrap: anywhere;
}
@media (max-width: 640px) {
    .user-msg { max-width: 86%; }
}
</style>
"""


def _render_user_message(text: str) -> None:
    """
    Draw a user turn as a right-aligned bubble with no avatar.

    The text is HTML escaped instead of going through Markdown, so a question
    containing *asterisks*, __init__.py, <b> or $ shows exactly as typed.
    Newlines become <br> after escaping, because Markdown ends an HTML block
    at the first blank line.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    safe = html.escape(normalized).replace("\n", "<br>")
    st.markdown(
        f'<div class="user-msg-row"><div class="user-msg">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def _render_assistant_message(text: str) -> None:
    """Draw a model turn as plain text in a bare container, which never draws an avatar."""
    with st.container():
        st.markdown(text)


tab_setup, tab_train, tab_eval, tab_chat = st.tabs(
    ["1. Setup", "2. Configure & Train", "3. Evaluate", "4. Chat"]
)

# ============================================================
# TAB 1: Setup - model + dataset selection
# ============================================================
with tab_setup:
    st.subheader("Select a base model")
    label_to_key = {entry.display_name: key for key, entry in MODEL_REGISTRY.items()}
    selected_label = st.selectbox("Base model", options=list(label_to_key.keys()))
    selected_key = label_to_key[selected_label]
    selected_model = MODEL_REGISTRY[selected_key]

    if selected_model.is_gated:
        # The app has no token field: downloads authenticate through whatever
        # login huggingface_hub already finds on the machine running the app.
        st.warning(
            f"{selected_model.display_name} is gated. Accept its license on its "
            f"Hugging Face page first. This app has no token field, so the machine "
            f"running it must already be logged in: run `hf auth login` (or "
            f"`huggingface_hub.login()` in Colab), or set the HF_TOKEN environment variable."
        )
    st.caption(
        f"~{selected_model.param_count_billions}B parameters · "
        f"repo: {selected_model.repo_id}"
    )

    st.subheader("Upload your dataset")
    uploaded_file = st.file_uploader("CSV or JSONL", type=["csv", "jsonl"])
    col1, col2 = st.columns(2)
    prompt_column = col1.text_input("Prompt column name", value="instruction")
    response_column = col2.text_input("Response column name", value="output")

    st.session_state["_setup"] = {
        "model_key": selected_key,
        "uploaded_file": uploaded_file,
        "prompt_column": prompt_column,
        "response_column": response_column,
    }

    st.divider()
    st.subheader("Or load an already-trained adapter")
    st.caption(
        "For testing a model fine-tuned in a previous session (e.g. saved to "
        "Google Drive) - uses the base model selected above, skips Setup/Train, "
        "and unlocks the Chat tab directly."
    )
    adapter_path_input = st.text_input(
        "Adapter path",
        placeholder="/content/drive/MyDrive/opentune-checkpoints/xlam-function-calling-001",
    )
    if st.button("Load Adapter"):
        if not adapter_path_input:
            st.error("Enter the adapter's path first.")
        elif not Path(adapter_path_input).exists():
            st.error(f"No adapter found at {adapter_path_input} - check the path.")
        else:
            with st.spinner("Loading adapter..."):
                try:
                    st.session_state["loaded_chat_model"] = load_finetuned_model(
                        base_model_id=selected_key, adapter_path=Path(adapter_path_input)
                    )
                    st.success(
                        f"Loaded {selected_model.display_name} + adapter. "
                        "Go to the Chat tab to test it."
                    )
                except Exception as e:
                    st.error(f"Failed to load adapter: {e}")


# ============================================================
# TAB 2: Configure & Train
# ============================================================
with tab_train:
    st.subheader("LoRA configuration")
    preset_choice = st.radio(
        "Preset",
        options=[p.value for p in LoRAPreset],
        horizontal=True,
        help="Fast/low-VRAM for quick iteration, Balanced as the default, "
        "Higher quality for larger datasets. Custom exposes raw values below.",
    )

    if preset_choice == LoRAPreset.CUSTOM.value:
        custom_r = st.number_input("Rank (r)", min_value=1, value=16)
        custom_alpha = st.number_input("Alpha", min_value=1, value=32)
    else:
        custom_r, custom_alpha = None, None

    st.subheader("Training settings")
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", min_value=1, value=3)
    batch_size = c2.number_input("Batch size", min_value=1, value=4)
    learning_rate = c3.number_input("Learning rate", min_value=0.0, value=2e-4, format="%.5f")

    if st.button("Start Fine-Tuning", type="primary"):
        setup = st.session_state.get("_setup", {})
        if not setup.get("uploaded_file"):
            st.error("Upload a dataset in the Setup tab first.")
        else:
            # "/tmp" is not a temp folder on Windows, and the client controls
            # the file name, so keep only its last path component.
            dataset_path = Path(tempfile.gettempdir()) / Path(setup["uploaded_file"].name).name
            dataset_path.write_bytes(setup["uploaded_file"].getvalue())

            lora_kwargs = {"preset": LoRAPreset(preset_choice)}
            if preset_choice == LoRAPreset.CUSTOM.value:
                lora_kwargs.update(r=custom_r, lora_alpha=custom_alpha)

            # pydantic's ValidationError is a ValueError. Catching only the
            # config construction turns a bad field value (e.g. a learning rate
            # of 0, which the number input allows) into a message, while errors
            # from training itself still surface unchanged.
            try:
                run_config = RunConfig(
                    run_name=f"run-{selected_model.display_name}".replace(" ", "-"),
                    base_model_id=st.session_state["_setup"]["model_key"],
                    dataset=DatasetConfig(
                        file_path=dataset_path,
                        prompt_column=setup["prompt_column"],
                        response_column=setup["response_column"],
                    ),
                    lora=LoRAConfig(**lora_kwargs),
                    training=TrainingConfig(
                        num_train_epochs=epochs,
                        per_device_train_batch_size=batch_size,
                        learning_rate=learning_rate,
                    ),
                )
            except ValueError as e:
                st.error(f"Invalid run configuration, training was not started:\n\n{e}")
            else:
                st.session_state["run_config"] = run_config

                with st.spinner("Fine-tuning in progress - this needs a GPU runtime..."):
                    result = run_training(
                        run_config,
                        checkpoints_dir=Path("checkpoints"),
                        logs_dir=Path("logs"),
                    )
                st.session_state["training_result"] = result
                st.success(f"Training complete. Final loss: {result.final_train_loss:.4f}")

    if st.session_state["training_result"]:
        r = st.session_state["training_result"]
        st.metric("Final train loss", f"{r.final_train_loss:.4f}")
        st.metric("Steps completed", r.num_steps_completed)

# ============================================================
# TAB 3: Evaluate
# ============================================================
with tab_eval:
    st.subheader("Compare fine-tuned vs. base model")
    if not st.session_state["training_result"]:
        st.info(
            "Evaluation needs a dataset to draw held-out examples from, so it's "
            "only available after a fresh training run in this session - not after "
            "loading an existing adapter. To evaluate an existing adapter, use the "
            "dedicated Colab evaluation script instead."
        )
    else:
        if st.button("Run Evaluation"):
            with st.spinner("Generating from both models and scoring - this takes a minute..."):
                run_config = st.session_state["run_config"]
                training_result = st.session_state["training_result"]

                if st.session_state["loaded_chat_model"] is None:
                    st.session_state["loaded_chat_model"] = load_finetuned_model(
                        base_model_id=run_config.base_model_id,
                        adapter_path=Path(training_result.adapter_path),
                    )
                finetuned_loaded = st.session_state["loaded_chat_model"]

                model_entry = get_model(run_config.base_model_id)
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True,
                )
                base_tokenizer = AutoTokenizer.from_pretrained(model_entry.repo_id)
                base_model_only = AutoModelForCausalLM.from_pretrained(
                    model_entry.repo_id, quantization_config=bnb_config, device_map="auto"
                )
                base_wrapped = LoadedModel(model=base_model_only, tokenizer=base_tokenizer, adapter_path="none")

                # The first rows of the validation slice that training held out
                # (same split and seed as build_dataset), so neither model was
                # trained on them.
                sample = held_out_examples(run_config.dataset, 3)
                eval_examples = [
                    {"instruction": ex["prompt"], "reference": ex["response"]} for ex in sample
                ]

                # Greedy decoding with room for a full JSON call, as in
                # scripts/colab_reevaluate_existing_adapter.py: sampling made two
                # runs disagree, and 60 tokens could cut a function call short.
                def generate_fn(instruction: str):
                    base_out = generate_response(base_wrapped, instruction, max_new_tokens=150, temperature=0)
                    finetuned_out = generate_response(finetuned_loaded, instruction, max_new_tokens=150, temperature=0)
                    return base_out, finetuned_out

                st.session_state["comparison_report"] = evaluate_models(
                    eval_examples, generate_fn=generate_fn, judge_fn=None
                )

        if st.session_state["comparison_report"]:
            report = st.session_state["comparison_report"]
            col1, col2 = st.columns(2)
            col1.metric("Base ROUGE-L", f"{report.avg_base_rouge_l:.3f}")
            col2.metric("Fine-tuned ROUGE-L", f"{report.avg_finetuned_rouge_l:.3f}")
            for ex in report.examples:
                with st.expander(ex.instruction):
                    st.write(f"**Base:** {ex.base_output}")
                    st.write(f"**Fine-tuned:** {ex.finetuned_output}")

# ============================================================
# TAB 4: Chat
# ============================================================
with tab_chat:
    st.subheader("Chat with the fine-tuned model")
    model_ready = (
        st.session_state["training_result"] is not None
        or st.session_state["loaded_chat_model"] is not None
    )
    if not model_ready:
        st.info("Complete a training run, or load an already-trained adapter in the Setup tab.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        st.markdown(CHAT_CSS, unsafe_allow_html=True)

        # st.chat_input inside a tab is drawn inline, not pinned to the bottom
        # of the page. The history lives in a container created above the input
        # so the new question and the spinner appear above the input box.
        history_area = st.container()
        user_msg = st.chat_input("Ask the fine-tuned model something...")

        with history_area:
            for role, msg in st.session_state["chat_history"]:
                if role == "user":
                    _render_user_message(msg)
                else:
                    _render_assistant_message(msg)

        if user_msg:
            st.session_state["chat_history"].append(("user", user_msg))
            with history_area:
                _render_user_message(user_msg)  # show the question straight away
                with st.container():
                    with st.spinner("Generating..."):
                        if st.session_state["loaded_chat_model"] is None:
                            st.session_state["loaded_chat_model"] = load_finetuned_model(
                                base_model_id=st.session_state["run_config"].base_model_id,
                                adapter_path=Path(st.session_state["training_result"].adapter_path),
                            )
                        response = generate_response(st.session_state["loaded_chat_model"], user_msg)
            st.session_state["chat_history"].append(("assistant", response))
            st.rerun()
