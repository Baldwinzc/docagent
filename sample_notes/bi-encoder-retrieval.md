# Bi-Encoder Retrieval

A **bi-encoder** encodes the query and each passage *independently* into vectors,
so passage vectors can be precomputed and searched with fast nearest-neighbour
lookup. It scales to millions of passages but, because query and passage never
interact during encoding, it is less precise than a cross-encoder.
