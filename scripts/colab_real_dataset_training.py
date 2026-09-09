# OpenTune - real dataset run: Salesforce xLAM function-calling data
# Self-contained: only needs the 6 backend .py files you already have
# (config.py, model_registry.py, data_pipeline.py, train.py, evaluate.py,
# inference.py). Does NOT need app.py, colab_app_walkthrough.py, or
# sample_dataset.jsonl - this is a script-based run, not the UI walkthrough.
#
# Before running: visit https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
# and click "Agree and access repository" - same one-time step as the gated
# models, required because this dataset is also gated.

!pip install -q transformers peft trl bitsandbytes accelerate datasets huggingface_hub rouge_score "nltk<3.10"
!pip uninstall -y -q torchao

from huggingface_hub import login
login()

import os
os.makedirs("/content/opentune", exist_ok=True)
os.chdir("/content/opentune")

print("Select the SIX backend files: config.py, model_registry.py, data_pipeline.py, train.py, evaluate.py, inference.py")
from google.colab import files
files.upload()

required = ["config.py", "model_registry.py", "data_pipeline.py", "train.py", "evaluate.py", "inference.py"]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    raise RuntimeError(f"Missing after upload: {missing}. Re-run this cell and select all six files.")
print("All six backend files confirmed present.\n")

import sys
from pathlib import Path
sys.path.insert(0, "/content/opentune")

# Mount Google Drive so checkpoints survive a runtime disconnect. Colab's
# local disk (/content/...) is wiped when the runtime disconnects or resets -
# the resume logic in train.py is useless if the checkpoint it would resume
# from no longer exists. This will prompt for Google auth the first time.
from google.colab import drive
drive.mount("/content/drive")

CHECKPOINTS_DIR = Path("/content/drive/MyDrive/opentune-checkpoints")
LOGS_DIR = Path("/content/drive/MyDrive/opentune-logs")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: Load and inspect the real dataset BEFORE assuming its shape
# ============================================================
import json
from datasets import load_dataset

print("=== Loading Salesforce/xlam-function-calling-60k ===")
raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

print("\nActual columns found:", raw.column_names)
print("\nFirst row, raw:")
print(raw[0])
print("\nIf this doesn't show query/tools/answers as you'd expect, STOP here")
print("and tell me exactly what printed above before continuing.\n")

# ============================================================
# STEP 2: Reformat into the instruction/output shape data_pipeline.py expects
# ============================================================
SUBSET_SIZE = 1200
subset = raw.shuffle(seed=42).select(range(min(SUBSET_SIZE, len(raw))))

def to_jsonl_str(value):
    """tools/answers may already be JSON strings or actual Python objects
    depending on how this dataset serializes them - handle both rather than
    assume one, since I couldn't directly inspect this dataset myself."""
    return value if isinstance(value, str) else json.dumps(value)

formatted_path = "/content/xlam_function_calling.jsonl"
with open(formatted_path, "w") as f:
    for row in subset:
        instruction = (
            f"Available tools:\n{to_jsonl_str(row['tools'])}\n\n"
            f"User request: {row['query']}"
        )
        output = to_jsonl_str(row["answers"])
        f.write(json.dumps({"instruction": instruction, "output": output}) + "\n")

print(f"Wrote {len(subset)} reformatted examples to {formatted_path}")
with open(formatted_path) as f:
    print("\nFirst reformatted example:")
    print(f.readline()[:500], "...")

# ============================================================
# STEP 3: Train on the real dataset
# ============================================================
from config import RunConfig, DatasetConfig, TrainingConfig
from train import run_training

run_config = RunConfig(
    run_name="xlam-function-calling-001",
    base_model_id="phi-3-mini",
    dataset=DatasetConfig(
        file_path=Path(formatted_path),
        validation_split=0.1,
        max_sequence_length=1024,  # longer than the smoke test - tool schemas take real space
    ),
    training=TrainingConfig(
        num_train_epochs=1,
        per_device_train_batch_size=2,
        checkpoint_every_n_steps=50,
    ),
)

print("\n=== TRAINING (this will take a while - real data this time) ===")
result = run_training(run_config, checkpoints_dir=CHECKPOINTS_DIR, logs_dir=LOGS_DIR)
print(f"Adapter saved to: {result.adapter_path}")
print(f"Final train loss: {result.final_train_loss}")
print(f"Steps completed: {result.num_steps_completed}")

# ============================================================
# STEP 4: Evaluate against the base model on real held-out examples
# ============================================================
print("\n=== EVALUATION ===")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model_registry import get_model
from inference import LoadedModel, load_finetuned_model, generate_response
from evaluate import evaluate_models
from data_pipeline import load_raw_examples

loaded_finetuned = load_finetuned_model(base_model_id="phi-3-mini", adapter_path=Path(result.adapter_path))

model_entry = get_model("phi-3-mini")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True,
)
base_tokenizer = AutoTokenizer.from_pretrained(model_entry.repo_id)
base_model_only = AutoModelForCausalLM.from_pretrained(
    model_entry.repo_id, quantization_config=bnb_config, device_map="auto"
)
base_only_wrapped = LoadedModel(model=base_model_only, tokenizer=base_tokenizer, adapter_path="none")

def generate_fn(instruction: str):
    base_out = generate_response(base_only_wrapped, instruction, max_new_tokens=120)
    finetuned_out = generate_response(loaded_finetuned, instruction, max_new_tokens=120)
    return base_out, finetuned_out

held_out_raw = load_raw_examples(run_config.dataset)[-5:]  # last 5, held out from training
eval_examples = [{"instruction": ex["prompt"], "reference": ex["response"]} for ex in held_out_raw]

report = evaluate_models(eval_examples, generate_fn=generate_fn, judge_fn=None)

print(f"\nAvg base ROUGE-L:       {report.avg_base_rouge_l:.3f}")
print(f"Avg fine-tuned ROUGE-L: {report.avg_finetuned_rouge_l:.3f}")
for ex in report.examples:
    print(f"\n--- {ex.instruction[:100]}...")
    print(f"  Base:       {ex.base_output[:200]!r}")
    print(f"  Fine-tuned: {ex.finetuned_output[:200]!r}")

print("\nIf everything above printed without error, you now have a real,")
print("portfolio-worthy fine-tuning result on real function-calling data.")
