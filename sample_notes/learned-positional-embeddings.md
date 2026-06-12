# Learned Positional Embeddings

Instead of fixed sinusoids, many models (e.g. BERT, GPT) use **learned positional
embeddings**: a trainable vector per absolute position, added to token embeddings.
They adapt to the data but are capped at the maximum position seen in training, so
they do not extrapolate to longer sequences without retraining or interpolation.
