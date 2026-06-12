# Causal Language Modeling (CLM)

**CLM** trains the model to predict the next token given only previous tokens
(left-to-right). It matches how text is generated at inference, so it is the
objective behind GPT-style generative models. Combined with a causal mask, every
position is trained in parallel.
