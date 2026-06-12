# WordPiece

**WordPiece** (used by BERT) also builds subwords by merging, but chooses the
merge that most increases the training corpus likelihood rather than raw
frequency. Continuation pieces are marked (e.g. `##ing`).
