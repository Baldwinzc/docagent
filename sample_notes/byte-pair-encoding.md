# Byte-Pair Encoding (BPE)

**BPE** builds a subword vocabulary by starting from characters and greedily
merging the most frequent adjacent pair, repeatedly, until a target vocabulary
size is reached. Frequent words end up as single tokens while rare words split
into pieces, bounding the vocabulary while covering any input.
