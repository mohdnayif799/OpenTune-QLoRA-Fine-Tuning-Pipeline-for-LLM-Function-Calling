# OpenTune - Module 4 (train.py) real GPU smoke test
# Run this in a Colab cell (Runtime -> Change runtime type -> T4 GPU).
# This is NOT a real training run - 5 examples, 5 steps, just proving the
# pipeline actually executes against real weights on a real GPU.

!pip install -q transformers peft trl bitsandbytes accelerate datasets huggingface_hub

from huggingface_hub import login
login()  # paste your HF token when prompted - needed even for ungated phi-3-mini's tokenizer config

import json
from pathlib import Path

# Adjust this to wherever you've placed the OpenTune files in this Colab session
import sys
sys.path.insert(0, "/content/opentune")

from config import RunConfig, DatasetConfig, TrainingConfig
from train import run_training

# Tiny synthetic dataset - just enough to prove the pipeline runs end to end
dummy_path = Path("/content/smoke_test.jsonl")
dummy_path.write_text("\n".join(
    json.dumps({"instruction": f"What is {i} + {i}?", "output": f"{i} + {i} = {i*2}"})
    for i in range(10)
))

run_config = RunConfig(
    run_name="smoke-test-001",
    base_model_id="phi-3-mini",  # smallest supported model - fastest smoke test
    dataset=DatasetConfig(file_path=dummy_path, validation_split=0.2, max_sequence_length=128),
    training=TrainingConfig(num_train_epochs=1, per_device_train_batch_size=2, checkpoint_every_n_steps=5),
)

result = run_training(
    run_config,
    checkpoints_dir=Path("/content/checkpoints"),
    logs_dir=Path("/content/logs"),
)

print("\n--- SMOKE TEST RESULT ---")
print(f"Adapter saved to: {result.adapter_path}")
print(f"Final train loss: {result.final_train_loss}")
print(f"Steps completed: {result.num_steps_completed}")
print("If this printed without error, train.py works end to end on real GPU weights.")
