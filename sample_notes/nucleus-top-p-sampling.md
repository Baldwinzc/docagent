# Nucleus (Top-p) Sampling

**Top-p / nucleus sampling** samples from the smallest set of tokens whose
cumulative probability exceeds p. The candidate set grows and shrinks with the
model's confidence, usually giving better open-ended generation than a fixed top-k.
