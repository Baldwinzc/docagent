# Prompt Tuning (Soft Prompts)

**Prompt tuning** learns a handful of continuous 'soft prompt' embeddings prepended
to the input, training only them. Unlike hand-written prompts these live in
embedding space, and the method becomes competitive with fine-tuning as model size
grows.
