# Cross-Encoder Reranker

A **cross-encoder** feeds the query and a candidate passage *together* through the
model and outputs a relevance score, letting them interact via attention. It is far
more accurate than a bi-encoder but too slow to run over a whole corpus, so it is
applied only to rerank a shortlist of candidates.
