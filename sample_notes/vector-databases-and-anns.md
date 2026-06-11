# Vector Databases and Approximate Nearest Neighbour Search

## What a vector database stores

A vector database stores, for each chunk of text, an **embedding** (a fixed-length
dense vector) together with the original text and **metadata** (source file, page,
line range). Retrieval is a nearest-neighbour search: embed the query, then find
the stored vectors closest to it, usually by **cosine similarity**. Metadata lets
the system attach precise provenance to every result and filter by source.

## Exact vs approximate search

**Exact** nearest-neighbour search compares the query against every stored vector.
It is accurate but scales linearly with the number of vectors, so it becomes slow
for large corpora. **Approximate nearest neighbour (ANN)** search trades a small
amount of recall for a large speedup by not examining every vector. For a personal
corpus exact search is fine; ANN matters at hundreds of thousands of vectors and up.

## Common ANN indexes

- **HNSW** (Hierarchical Navigable Small World) builds a multi-layer graph and
  greedily walks it toward the query's neighbours. It gives high recall at low
  latency and supports incremental inserts, at the cost of memory.
- **IVF** (inverted file) clusters vectors and searches only the clusters nearest
  the query, trading recall for speed via how many clusters it probes.
- **Product quantization** compresses vectors so more of the index fits in memory,
  at some loss of precision.

## Chunking

Documents are split into **chunks** before embedding because embeddings summarise a
bounded span of text, and retrieval returns whole chunks. Chunks that are too large
dilute the embedding and waste context; too small lose surrounding meaning. A common
approach uses a fixed size with an **overlap** between consecutive chunks so a fact
spanning a boundary is not lost. Each chunk keeps a stable id and its provenance.

## Sparse alongside dense

A vector index handles dense (semantic) retrieval, but exact-keyword matching is
better served by a separate **inverted index** (for BM25). Systems that need both
keep the dense index in the vector database and the sparse index alongside it, then
fuse the two result lists.
