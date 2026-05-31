# docagent architecture

docagent is built on LangGraph as a two-layer state graph.

## Top layer: intent routing

The entry node is the `intent_router`. It uses an LLM with structured output to
classify the incoming question as either `in_scope` or `out_of_scope`.

- `in_scope` questions are forwarded to the response agent.
- `out_of_scope` questions end immediately with a polite refusal.

## Bottom layer: the response agent

The response agent is a tool-calling loop with three moving parts:

- `llm_call`: the LLM picks the next tool to call.
- `environment`: executes the tool and returns the result.
- `should_continue`: ends the loop when the agent calls a terminal tool
  (`Answer` or `Question`), otherwise routes back to run more tools.

## Tools

- `search_docs(query, k)`: semantic search over the knowledge base.
- `list_sources()`: list the documents currently indexed.
- `Answer(answer, citations)`: the terminal tool that returns the grounded answer.
- `Question(content)`: ask the user for clarification.

## Storage and embeddings

Chunks are stored in a persistent **Chroma** vector store on local disk (default
directory `./chroma_db`). Embeddings are produced by a local
**sentence-transformers** model (default `all-MiniLM-L6-v2`).
