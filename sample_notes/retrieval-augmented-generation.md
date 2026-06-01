# Retrieval-Augmented Generation (RAG)

## The idea

A language model's parametric memory is fixed at training time, hard to update,
and prone to hallucination on niche facts. **Retrieval-Augmented Generation**
adds a non-parametric memory: a retriever fetches relevant passages from an
external corpus, and the generator conditions its answer on those passages. This
makes answers updatable (swap the corpus, not the weights) and groundable (the
answer can cite where it came from).

## Pipeline

1. **Index**: split documents into chunks and store them (vectors for dense
   retrieval, and/or an inverted index for BM25).
2. **Retrieve**: for a query, fetch the top-k most relevant chunks.
3. **Generate**: give the query plus retrieved chunks to the LLM and ask it to
   answer using only that context.

## Grounding and citations

Because the answer is conditioned on specific retrieved chunks, each claim can be
traced to a source. A robust system goes further and **verifies** that the
citations an LLM emits actually correspond to retrieved chunks, dropping any that
do not — otherwise "citations" are just text the model produced.

## Agentic RAG

Plain RAG retrieves once and answers. **Agentic RAG** lets the model decide: it
can search, inspect the results, reformulate the query, search again, and only
answer once it has enough evidence — or decline if the corpus does not cover the
question. The retrieval quality (hybrid search + reranking, see the retrieval
note) largely determines how well this loop works.
