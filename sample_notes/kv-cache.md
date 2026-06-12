# KV Cache

During autoregressive generation, the keys and values for earlier tokens do not
change, so a **KV cache** stores them and only computes attention for the new
token each step. This turns per-step cost from quadratic to linear in sequence
length, at the cost of memory that grows with context length and batch size.
