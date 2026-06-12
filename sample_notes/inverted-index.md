# Inverted Index

An **inverted index** maps each term to the list of documents (and positions)
containing it. Sparse retrieval scores only documents that share a query term, so
lookups are fast over huge corpora — the data structure behind BM25 search.
