# SentencePiece and the Unigram Model

**SentencePiece** tokenizes raw text directly (treating spaces as a symbol), so it
is language-agnostic and reversible. Its **Unigram** algorithm starts from a large
candidate vocabulary and prunes it to maximize likelihood, keeping multiple
segmentations possible.
