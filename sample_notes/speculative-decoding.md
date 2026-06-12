# Speculative Decoding

**Speculative decoding** uses a small fast 'draft' model to propose several tokens
that the large model then verifies in one pass, accepting the longest correct
prefix. It speeds up generation with identical output distribution to the large
model alone.
