# Parametric vs Non-Parametric Memory

**Parametric memory** is knowledge baked into a model's weights — fixed at
training, hard to update, and prone to hallucination on niche facts.
**Non-parametric memory** is an external corpus the model retrieves from at
inference; it is updatable (swap the corpus, not the weights) and citable, which is
why RAG pairs the two.
