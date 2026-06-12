# Product Quantization (PQ)

**Product quantization** splits each vector into sub-vectors and replaces each with
the id of the nearest centroid from a small codebook, compressing vectors to a few
bytes. Far more of the index fits in memory and distances are estimated from
codes, at some loss of precision; PQ is often combined with IVF.
