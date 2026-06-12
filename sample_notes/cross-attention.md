# Cross-Attention

**Cross-attention** lets one sequence attend to another: queries come from the
decoder while keys and values come from the encoder's output. It is how an
encoder-decoder model (translation, summarization) conditions generation on the
source. Self-attention, by contrast, draws Q, K and V from the same sequence.
