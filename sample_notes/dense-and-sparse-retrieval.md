# Dense and Sparse Retrieval

## Sparse retrieval (BM25)

Classic information retrieval represents text as sparse bag-of-words vectors and
scores documents with **BM25**, a TF-IDF-style ranking function with term
saturation and length normalisation. BM25 is strong on exact keyword overlap and
rare terms (names, identifiers, error codes), needs no training, and is cheap to
run, but it cannot match paraphrases that share no words.

## Dense retrieval (embeddings)

Dense retrieval encodes each passage into a fixed-length vector with a
Transformer encoder, then finds nearest neighbours to the query vector by cosine
similarity. It captures semantic similarity even when wording differs, but it can
miss exact terms and is sensitive to the embedding model's training domain.

## Hybrid retrieval and fusion

Dense and sparse retrieval have complementary failure modes, so combining them is
usually better than either alone. A simple, robust way to merge two ranked lists
is **Reciprocal Rank Fusion (RRF)**: each document's score is the sum of
`1 / (k + rank)` over the lists it appears in, so documents ranked highly by
either method rise to the top without needing calibrated scores.

## Reranking

Bi-encoder retrieval (separate query/passage encoders) is fast but approximate. A
**cross-encoder reranker** reads the query and a candidate passage together and
outputs a relevance score; it is far more accurate but too slow to run over a
whole corpus, so it is applied only to a shortlist of fused candidates. A
relevance threshold on the reranker score also gives a principled way to return
nothing when no passage is actually relevant.
