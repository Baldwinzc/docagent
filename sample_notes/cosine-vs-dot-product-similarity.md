# Cosine vs Dot-Product Similarity

**Cosine similarity** compares vector *direction* (magnitude-invariant);
**dot product** also rewards magnitude. With L2-normalized vectors the two rank
identically. The choice must match how the embedding model was trained, or ranking
quality suffers.
