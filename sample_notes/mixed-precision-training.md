# Mixed-Precision Training

**Mixed precision** stores and computes most tensors in 16-bit (fp16/bf16) while
keeping a 32-bit master copy of the weights and using loss scaling to avoid
underflow. It roughly halves memory and speeds up training on modern accelerators
with little or no accuracy loss.
