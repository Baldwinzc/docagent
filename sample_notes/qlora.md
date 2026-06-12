# QLoRA

**QLoRA** fine-tunes a base model that is **quantized to 4-bit** while training
LoRA adapters in higher precision on top. It slashes the memory needed to
fine-tune large models, making single-GPU adaptation of big models feasible with
little quality loss.
