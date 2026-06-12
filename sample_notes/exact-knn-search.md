# Exact Nearest-Neighbour Search

**Exact kNN** compares the query against every stored vector and returns the true
closest ones. It is perfectly accurate but cost grows linearly with the number of
vectors, so it is fine for thousands but slow for very large corpora.
