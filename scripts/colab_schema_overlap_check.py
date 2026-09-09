# OpenTune - schema-overlap check: does the 120-example eval set test genuinely
# unseen TOOLS, or just unseen phrasing of tools already seen in training?
# No GPU needed - dataset inspection only, a few seconds to run.

import json
from datasets import load_dataset

raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
subset = raw.shuffle(seed=42).select(range(1200))  # same seed as training/eval

def tool_names(row):
    answers = row["answers"] if isinstance(row["answers"], list) else json.loads(row["answers"])
    return {a["name"] for a in answers if isinstance(a, dict) and "name" in a}

# Same 90/10 split logic as data_pipeline.split_train_val (seed=42 default)
import random
indices = list(range(len(subset)))
random.Random(42).shuffle(indices)
val_size = int(len(indices) * 0.1)
val_idx, train_idx = indices[:val_size], indices[val_size:]

train_tools = set()
for i in train_idx:
    train_tools |= tool_names(subset[i])

eval_tools = set()
for i in val_idx:
    eval_tools |= tool_names(subset[i])

overlap = train_tools & eval_tools
overlap_pct = len(overlap) / len(eval_tools) if eval_tools else 0

print(f"Distinct tools in training split:   {len(train_tools)}")
print(f"Distinct tools in eval split:       {len(eval_tools)}")
print(f"Tools in eval also seen in training: {len(overlap)} ({overlap_pct:.0%})")
print(f"Tools in eval NEVER seen in training: {len(eval_tools - train_tools)}")
print()
if overlap_pct > 0.8:
    print("=> High overlap. Honest claim: 'reliably calls a KNOWN toolset on novel")
    print("   phrasing/queries' - not 'generalizes to unseen tools'. Word the README accordingly.")
else:
    print("=> Meaningful fraction of eval tools were never seen in training - genuine")
    print("   tool-generalization evidence, not just phrasing generalization.")
