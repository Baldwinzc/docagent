# AGENTS.md

Guidance for AI agents working in this repository. See `CLAUDE.md` for the full
module-by-module guide; this is the short version.

## What this is

docagent: an agentic-RAG document QA agent on LangGraph. Ingests local docs into
Chroma, answers questions with forced citations.

## Setup

```bash
conda create -n docagent python=3.11 -c conda-forge && conda activate docagent
pip install -e .
```

## Where things live

- Graph: `src/docagent/agent.py` (`docagent` compiled, `overall_workflow` builder)
- Tools: `src/docagent/tools/retrieval_tools.py`
- Ingest / ask CLIs: `src/docagent/ingest.py`, `src/docagent/ask.py`
- Tests: `tests/` (`test_retrieval.py` needs no key; `test_response.py` does)

## Rules

- Answer only from retrieved content; always cite sources.
- Embeddings run locally (no API key); only the answer LLM needs a key.
- Run `python tests/run_all_tests.py` before claiming tests pass.
