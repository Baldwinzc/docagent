# Rotary Position Embeddings (RoPE)

**RoPE** encodes position by rotating the query and key vectors by an angle that
depends on their position before the dot product. This makes attention scores
depend on the *relative* offset between tokens, generalizes better to longer
contexts than absolute embeddings, and is widely used in modern LLMs (LLaMA, etc.).
