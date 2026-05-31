# About docagent

docagent is a local document question-answering agent. You point it at a folder
of your own files, it builds a searchable knowledge base, and then it answers
questions about those files — always citing the exact sources it used.

## What makes it different from plain RAG

Most retrieval-augmented-generation demos do a single retrieval and then
generate an answer. docagent runs an **agentic RAG loop**: the agent decides how
many times to search, reformulates the query when the first results are weak, and
only produces a final answer once it has gathered enough evidence.

## Key features

- **Agentic retrieval**: the agent can call `search_docs` multiple times with
  different queries before answering.
- **Forced citations**: the final answer is produced through an `Answer` tool
  that requires a list of source citations, so no claim ships ungrounded.
- **Intent routing**: an up-front router classifies each question as `in_scope`
  or `out_of_scope`; out-of-scope questions are politely declined without wasting
  a retrieval or an answer.
- **Local embeddings**: documents are embedded with sentence-transformers, which
  runs locally and needs no API key.

## Who it is for

Anyone who wants to ask natural-language questions over a private set of
documents — product docs, research notes, a handbook — without sending the
documents themselves to a third-party indexing service.
