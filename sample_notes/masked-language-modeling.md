# Masked Language Modeling (MLM)

**MLM** randomly masks a fraction of tokens (about 15% in BERT) and trains the
model to predict them from both left and right context. Conditioning on both sides
yields deep **bidirectional** representations, which suit understanding tasks but
make the model less natural for free-form generation.
