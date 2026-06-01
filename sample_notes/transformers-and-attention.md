# Transformers and Attention

## Motivation

Recurrent models (RNNs, LSTMs) process a sequence token by token, so computation
cannot be parallelised across positions and long-range dependencies are hard to
learn. The Transformer replaces recurrence entirely with **attention**, which
relates any two positions in a sequence in a single step.

## Scaled dot-product attention

Attention maps a query and a set of key-value pairs to an output. For queries Q,
keys K, and values V, the output is a weighted sum of the values:

    Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V

The scaling factor `1 / sqrt(d_k)` keeps the dot products from growing large in
high dimensions, which would otherwise push the softmax into regions with tiny
gradients.

## Multi-head attention

Instead of one attention function, the Transformer projects Q, K, V into `h`
lower-dimensional subspaces, runs attention in each, and concatenates the
results. Different heads can attend to different kinds of relationships (syntax,
coreference, position) in parallel.

## Why it matters for retrieval systems

The same attention idea underlies modern **text embeddings**: a Transformer
encoder turns a passage into a dense vector whose geometry captures meaning, which
is exactly what dense retrieval relies on (see the note on dense and sparse
retrieval). Encoder-only Transformers (like BERT-style models) are the usual
backbone for embedding and reranking models.
