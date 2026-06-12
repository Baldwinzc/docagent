# Learned Sparse Retrieval (SPLADE)

**Learned sparse retrieval** (e.g. SPLADE) uses a Transformer to predict
term-importance weights over the vocabulary, including expansion terms the document
did not literally contain. The result is a sparse vector usable with an inverted
index, combining BM25's efficiency and exact-match strengths with learned
semantics.
