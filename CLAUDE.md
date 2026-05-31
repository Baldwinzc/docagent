# docagent — guide for AI coding assistants

## Overview

docagent is an **agentic-RAG document question-answering agent** built on
LangGraph. It ingests local documents (Markdown / text / PDF) into a Chroma
vector store and answers questions about them, always returning citations.

## Environment setup

```bash
conda create -n docagent python=3.11 -c conda-forge
conda activate docagent
pip install -e .
```

The answer LLM needs an API key (`OPENAI_API_KEY` in `.env`), or set
`LLM_MODEL=ollama:llama3.1` for a fully local setup. Embeddings always run
locally and need no key.

## Key modules

- `src/docagent/agent.py` — the LangGraph graph. Compiled graph object:
  `docagent`; uncompiled builder (used by tests): `overall_workflow`. Two layers:
  `intent_router` (in_scope / out_of_scope) → `response_agent` (the RAG loop:
  `llm_call` → `should_continue` → `environment`).
- `src/docagent/ingest.py` — ingestion CLI (`python -m docagent.ingest --path ...`).
- `src/docagent/ask.py` — query CLI (`python -m docagent.ask "..."`).
- `src/docagent/vectorstore.py` — shared embeddings + Chroma backend, imported by
  both ingest and the retrieval tools so they stay in sync.
- `src/docagent/tools/retrieval_tools.py` — `search_docs`, `list_sources`,
  `Answer` (terminal tool that forces citations), `Question`.

## Running

```bash
python -m docagent.ingest --path ./sample_docs   # build the knowledge base
python -m docagent.ask "What vector store does docagent use?"
langgraph dev                                     # LangGraph Studio
```

## Testing

```bash
python tests/run_all_tests.py         # local retrieval tests (no key)
python tests/run_all_tests.py --all   # + LLM end-to-end (needs key)
```

## Conventions

- The agent must answer **only** from retrieved content and cite its sources.
- `Answer` and `Question` are terminal tools — calling them ends the loop.
- Keep embeddings local; only the answer step may call an external LLM.
