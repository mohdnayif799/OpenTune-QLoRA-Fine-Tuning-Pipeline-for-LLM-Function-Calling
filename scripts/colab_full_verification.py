# OpenTune - full GPU verification: train.py + evaluate.py + inference.py
# ONE script, ONE session. Uses files.upload() instead of dragging into the
# side panel - that upload always lands in the notebook's current working
# directory, so there's no way for it to end up in the wrong folder.

!pip install -q transformers peft trl bitsandbytes accelerate datasets huggingface_hub rouge_score "nltk<3.10"

# Colab's base image ships torchao 0.10.0 pre-installed, which is too old for
# current peft's internal version check - peft raises ImportError from its
# own capability-detection code the moment PeftModel.from_pretrained() runs,
# even though this project never uses torchao (bitsandbytes is the only
# quantization backend OpenTune uses - see design doc Section 13). Removing
# it rather than upgrading it, since there's no real use for it here either way.
!pip uninstall -y -q torchao

from huggingface_hub import login
login()

import os
os.makedirs("/content/opentune", exist_ok=True)
os.chdir("/content/opentune")

print("Select ALL SIX files at once: config.py, model_registry.py, data_pipeline.py, train.py, evaluate.py, inference.py")
from google.colab import files
files.upload()

import sys
sys.path.insert(0, "/content/opentune")

required = ["config.py", "model_registry.py", "data_pipeline.py", "train.py", "evaluate.py", "inference.py"]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    raise RuntimeError(f"Missing after upload: {missing}. Re-run this cell and select all six files.")
print("All six files confirmed present in /content/opentune - continuing.\n")

import json
from pathlib import Path
from config import RunConfig, DatasetConfig, TrainingConfig
from train import run_training

dummy_path = Path("/content/smoke_test.jsonl")
dummy_path.write_text("\n".join(
    json.dumps({"instruction": f"What is {i} + {i}?", "output": f"{i} + {i} = {i*2}"})
    for i in range(10)
))

run_config = RunConfig(
    run_name="smoke-test-001",
    base_model_id="phi-3-mini",
    dataset=DatasetConfig(file_path=dummy_path, validation_split=0.2, max_sequence_length=128),
    training=TrainingConfig(num_train_epochs=1, per_device_train_batch_size=2, checkpoint_every_n_steps=5),
)

print("=== TRAINING ===")
result = run_training(run_config, checkpoints_dir=Path("/content/checkpoints"), logs_dir=Path("/content/logs"))
print(f"Adapter saved to: {result.adapter_path}")
print(f"Final train loss: {result.final_train_loss}")
print(f"Steps completed: {result.num_steps_completed}")

print("\n=== EVALUATION + INFERENCE ===")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model_registry import get_model
from inference import LoadedModel, load_finetuned_model, generate_response
from evaluate import evaluate_models

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
    base_out = generate_response(base_only_wrapped, instruction, max_new_tokens=40)
    finetuned_out = generate_response(loaded_finetuned, instruction, max_new_tokens=40)
    return base_out, finetuned_out

eval_examples = [
    {"instruction": "What is 3 + 3?", "reference": "3 + 3 = 6"},
    {"instruction": "What is 7 + 7?", "reference": "7 + 7 = 14"},
]
report = evaluate_models(eval_examples, generate_fn=generate_fn, judge_fn=None)

print(f"Avg base ROUGE-L:       {report.avg_base_rouge_l:.3f}")
print(f"Avg fine-tuned ROUGE-L: {report.avg_finetuned_rouge_l:.3f}")
for ex in report.examples:
    print(f"\nInstruction: {ex.instruction}")
    print(f"  Base:       {ex.base_output!r}")
    print(f"  Fine-tuned: {ex.finetuned_output!r}")

chat_reply = generate_response(loaded_finetuned, "What is 10 + 10?")
print(f"\nChat test - fine-tuned model replied: {chat_reply!r}")

print("\nIf everything above printed without error, train.py, evaluate.py, and")
print("inference.py all work end to end on real GPU weights.")
