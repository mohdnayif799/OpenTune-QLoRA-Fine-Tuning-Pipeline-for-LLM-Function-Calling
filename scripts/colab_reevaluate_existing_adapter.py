# OpenTune - RE-EVALUATION ONLY of the already-trained xLAM adapter.
# Does NOT retrain. Loads the adapter already saved to Google Drive from
# your last successful run, reconstructs the same 120 held-out examples
# (same seed, same subset - deterministic), and re-evaluates using the
# fixed generate_response() and the new function-calling-specific metrics
# instead of ROUGE-L alone.

!pip install -q transformers peft trl bitsandbytes accelerate datasets huggingface_hub rouge_score "nltk<3.10"
!pip uninstall -y -q torchao

from huggingface_hub import login
login()

from google.colab import drive
drive.mount("/content/drive")

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

import sys
sys.path.insert(0, "/content/opentune")

# ============================================================
# Reconstruct the EXACT SAME 1200-example subset and 120-example
# held-out split used during training - same seed, same logic, so
# this is evaluating on genuinely unseen data, not a fresh random slice.
# ============================================================
import json
from datasets import load_dataset

print("=== Reconstructing the same dataset split used in training ===")
raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
subset = raw.shuffle(seed=42).select(range(1200))  # same seed as before

def to_jsonl_str(value):
    return value if isinstance(value, str) else json.dumps(value)

formatted_path = "/content/xlam_function_calling.jsonl"
with open(formatted_path, "w") as f:
    for row in subset:
        instruction = f"Available tools:\n{to_jsonl_str(row['tools'])}\n\nUser request: {row['query']}"
        output = to_jsonl_str(row["answers"])
        f.write(json.dumps({"instruction": instruction, "output": output}) + "\n")

from pathlib import Path
from config import DatasetConfig
from data_pipeline import load_raw_examples, split_train_val

dataset_config = DatasetConfig(file_path=Path(formatted_path), validation_split=0.1, max_sequence_length=1024)
all_examples = load_raw_examples(dataset_config)
_, held_out = split_train_val(all_examples, dataset_config.validation_split)  # same seed=42 default
print(f"Reconstructed {len(held_out)} held-out examples (should be 120)")

# ============================================================
# Load the ALREADY-TRAINED adapter from Drive - no retraining
# ============================================================
print("\n=== Loading the already-trained adapter from Drive ===")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model_registry import get_model
from inference import LoadedModel, load_finetuned_model, generate_response
from evaluate import evaluate_models

ADAPTER_PATH = Path("/content/drive/MyDrive/opentune-checkpoints/xlam-function-calling-001")
if not ADAPTER_PATH.exists():
    raise RuntimeError(f"No adapter found at {ADAPTER_PATH} - check the path matches your Drive.")

loaded_finetuned = load_finetuned_model(base_model_id="phi-3-mini", adapter_path=ADAPTER_PATH)

model_entry = get_model("phi-3-mini")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True,
)
base_tokenizer = AutoTokenizer.from_pretrained(model_entry.repo_id)
base_model_only = AutoModelForCausalLM.from_pretrained(
    model_entry.repo_id, quantization_config=bnb_config, device_map="auto"
)
base_wrapped = LoadedModel(model=base_model_only, tokenizer=base_tokenizer, adapter_path="none")

# ============================================================
# Real evaluation: all 120 held-out examples, fixed generation, real metrics
# ============================================================
print(f"\n=== Evaluating on all {len(held_out)} held-out examples (this will take a while) ===")

def generate_fn(instruction: str):
    # temperature=0 forces deterministic greedy decoding (do_sample=False)
    # instead of generate_response's default 0.7 sampling. Evaluation needs
    # reproducible output; app.py's chat tab correctly keeps the sampling
    # default since variety is desirable there, not here.
    base_out = generate_response(base_wrapped, instruction, max_new_tokens=150, temperature=0)
    finetuned_out = generate_response(loaded_finetuned, instruction, max_new_tokens=150, temperature=0)
    return base_out, finetuned_out

eval_examples = [{"instruction": ex["prompt"], "reference": ex["response"]} for ex in held_out]
report = evaluate_models(eval_examples, generate_fn=generate_fn, judge_fn=None)

print("\n" + "=" * 60)
print("RESULTS (120 held-out examples, base vs. fine-tuned)")
print("=" * 60)
print(f"{'Metric':<25} {'Base':>10} {'Fine-tuned':>12}")
print(f"{'JSON validity rate':<25} {report.base_json_validity_rate:>10.1%} {report.finetuned_json_validity_rate:>12.1%}")
print(f"{'Tool selection accuracy':<25} {report.base_tool_accuracy:>10.1%} {report.finetuned_tool_accuracy:>12.1%}")
print(f"{'Exact match rate':<25} {report.base_exact_match_rate:>10.1%} {report.finetuned_exact_match_rate:>12.1%}")
print(f"{'ROUGE-L (secondary)':<25} {report.avg_base_rouge_l:>10.3f} {report.avg_finetuned_rouge_l:>12.3f}")

print("\n--- 5 example outputs (now correctly stripped of echoed prompts) ---")
for ex in report.examples[:5]:
    print(f"\nQuery excerpt: ...{ex.instruction[-150:]}")
    print(f"  Reference:  {ex.reference[:200]}")
    print(f"  Base:       {ex.base_output[:200]!r}")
    print(f"  Fine-tuned: {ex.finetuned_output[:200]!r}")

print("\nThese are the numbers worth putting in a README/resume - decomposed,")
print("objective, and defensible, unlike a single ROUGE score would be.")
