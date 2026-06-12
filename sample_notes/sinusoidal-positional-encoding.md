# Sinusoidal Positional Encoding

Attention is permutation-invariant, so order must be injected explicitly. The
original Transformer adds **sinusoidal positional encodings** — fixed sine/cosine
functions of position at different frequencies — to the token embeddings. Because
they are deterministic, they extrapolate to sequence lengths unseen in training
and require no parameters.
