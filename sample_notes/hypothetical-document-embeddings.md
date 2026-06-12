# HyDE

**HyDE** (Hypothetical Document Embeddings) asks an LLM to draft a hypothetical
answer to the query, then embeds *that* and retrieves passages similar to it. The
fabricated answer is closer in embedding space to real relevant passages than the
short query is, improving zero-shot dense retrieval.
