# LoRA

**LoRA** freezes the base weights and learns a low-rank update (two small matrices
whose product is added to a weight matrix). It trains well under 1% of the
parameters, matches full fine-tuning on many tasks, and the small deltas can be
merged in or swapped per task.
