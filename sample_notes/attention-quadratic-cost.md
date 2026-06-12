# Quadratic Cost of Attention

Self-attention compares every token with every other token, so compute and memory
scale **O(n^2)** with sequence length n. This is the main bottleneck for long
contexts and motivates KV caching, sparse/linear attention variants, and
IO-aware kernels like FlashAttention.
