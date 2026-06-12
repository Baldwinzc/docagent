# Position-wise Feed-Forward Network

After attention, each Transformer block applies a **position-wise feed-forward
network (FFN)**: two linear layers with a non-linearity between them, applied
independently to every position. It typically expands to ~4x the model dimension
and back. Attention mixes information *across* positions; the FFN transforms each
position's representation on its own. Most of a model's parameters live in FFNs.
