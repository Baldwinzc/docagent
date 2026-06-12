# Reciprocal Rank Fusion (RRF)

**RRF** merges several ranked lists by summing `1/(k + rank)` for each document
across the lists it appears in. It needs no score calibration between systems and
lets a document ranked highly by *any* method rise to the top — a simple, robust
way to combine dense and sparse results.
