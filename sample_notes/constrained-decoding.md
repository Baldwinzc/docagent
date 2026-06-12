# Constrained Decoding

**Constrained decoding** forces outputs to satisfy a structure — valid JSON, a
regex, a grammar, or a fixed set of choices — by masking out tokens that would
violate the constraint at each step. It is how tool-calling and schema-conformant
outputs are guaranteed.
