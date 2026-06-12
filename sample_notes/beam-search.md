# Beam Search

**Beam search** keeps the top-k partial sequences (beams) at each step and expands
them, returning the highest-scoring complete sequence. It explores more of the
space than greedy decoding and suits tasks with a 'correct' output (translation),
but can be bland for open-ended generation and needs length normalization.
