# HNSW Index

**HNSW** builds a multi-layer 'navigable small world' graph and greedily walks from
a coarse top layer down to fine layers toward the query's neighbours. It offers
high recall at low latency and supports incremental inserts, at the cost of memory.
