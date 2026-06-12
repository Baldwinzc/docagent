# Context-Window Management

The **context window** caps how many tokens (prompt + retrieved chunks + output)
fit per call. RAG must budget it: retrieve enough evidence without overflowing,
since stuffing too many or low-relevance chunks wastes tokens, raises cost, and can
bury the answer ('lost in the middle').
