# Span Corruption

T5's pre-training objective **masks contiguous spans** of tokens and trains the
model to regenerate the missing spans as a single target sequence. Framing the
task as text-to-text fits its encoder-decoder design and unifies many tasks under
one objective.
