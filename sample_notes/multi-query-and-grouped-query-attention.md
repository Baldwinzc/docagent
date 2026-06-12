# Multi-Query and Grouped-Query Attention

**Multi-query attention (MQA)** shares a single key/value head across all query
heads, shrinking the KV cache and speeding decoding at a small quality cost.
**Grouped-query attention (GQA)** is the middle ground: several query heads share
each KV head, recovering most of the quality while keeping the cache small.
