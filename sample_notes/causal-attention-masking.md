# Causal Attention Masking

In a decoder (autoregressive) model, each position may only attend to itself and
earlier positions. This is enforced with a **causal mask** that sets attention
scores to future positions to negative infinity before the softmax, so they get
zero weight. It is what lets the model be trained on all positions in parallel
while still predicting the next token left-to-right.
