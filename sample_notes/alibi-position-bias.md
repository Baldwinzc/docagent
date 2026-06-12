# ALiBi Position Bias

**ALiBi** (Attention with Linear Biases) adds a distance-proportional penalty
directly to the attention scores instead of using positional embeddings: tokens
farther apart are biased toward lower attention. It is simple and extrapolates to
sequences much longer than those seen in training.
