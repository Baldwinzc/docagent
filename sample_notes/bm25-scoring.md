# BM25 Scoring

**BM25** scores query-document matches with TF-IDF-style weights plus two
refinements: **term-frequency saturation** (extra occurrences help less and less)
and **length normalization** (long documents are not unfairly favored). It is a
strong, training-free sparse baseline, excellent on exact and rare terms.
