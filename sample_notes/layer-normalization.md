# Layer Normalization

**Layer normalization** normalizes the activations across the feature dimension
of a single token (its mean and variance over that token's vector), then rescales
with learned gain and bias. Unlike batch norm it does not depend on other examples
in the batch, so it works with variable-length sequences and small batches — the
reason Transformers use it. **Pre-norm** (normalize before the sub-layer) is more
stable to train than post-norm for deep models.
