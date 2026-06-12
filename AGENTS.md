# AGENTS.md

Guidance for AI coding assistants working in this repository.

## What this is

citelocal-agent is an **agentic-RAG document question-answering agent** built on
LangGraph. It ingests local documents (Markdown / text / PDF) into a Chroma
vector store and answers questions about them, always returning citations.

## Setup

```bash
conda create -n citelocal-agent python=3.11 -c conda-forge && conda activate citelocal-agent
pip install -e ".[dev]"
```

The answer LLM needs an API key (`OPENAI_API_KEY` in `.env`), or any
OpenAI-compatible endpoint via `OPENAI_BASE_URL` / `LLM_MODEL` (e.g.
`LLM_MODEL=ollama:llama3.1` for a fully local setup). Embeddings always run
locally and need no key.

## Where things live

- `src/citelocal_agent/agent.py` — the LangGraph graph, built by the `build_agent(config)`
  factory (nothing initialised at import). `intent_router` classifies scope **and
  complexity**, then routes: `in_scope + simple` → `response_agent` (the single RAG
  loop `llm_call` → `should_continue` → `environment`, built by
  `build_research_loop`); `in_scope + complex` → `orchestrator` (multi-agent).
  `get_default_agent()` is single-shot; `get_chat_agent()` compiles with an
  `InMemorySaver` checkpointer for multi-turn memory (invoke with a `thread_id`).
  Recursion budgets are per-path, from `Configuration` (wrapper nodes invoke each
  inner graph with its own limit).
- `src/citelocal_agent/orchestrator.py` — the complex path: `planner` decomposes the
  question → parallel `researcher`s (each reuses the same retrieval loop, via the
  `Send` API) → `verifier` (per-sentence entailment) → `synthesizer` (one final
  `Answer`). Researchers write to `sub_results` (not `messages`) so parallel
  branches merge via `operator.add`.
- `src/citelocal_agent/verify.py` — `verify_claims()`: split an answer into sentences and
  check each is entailed by the retrieved evidence (backend: NLI cross-encoder /
  LLM judge / injected `entail_fn` / off).
- `src/citelocal_agent/retriever.py` — hybrid dense + BM25 (persistent bm25s index) → RRF
  → cross-encoder rerank → relevance threshold. `vectorstore.py` is the shared
  embeddings + Chroma backend.
- `src/citelocal_agent/tools/retrieval_tools.py` — `search_docs`, `list_sources`,
  `Answer` (terminal tool that forces citations), `Question`.
- `src/citelocal_agent/web.py` — FastAPI app (`/api/ask`, SSE `/api/ask/stream`, `/health`,
  multi-collection, per-thread sessions); `security.py` — framework-free API-key
  check + rate limiter wired in as dependencies. `Dockerfile` serves it.
- Ingest / ask / chat CLIs: `src/citelocal_agent/ingest.py`, `src/citelocal_agent/ask.py`,
  `src/citelocal_agent/chat.py` (multi-turn REPL).
- Eval: `src/citelocal_agent/eval/` (`data/qa_cases.jsonl` dataset + `run_eval.py`).

## Running

```bash
python -m citelocal_agent.ingest --path ./sample_notes --reset   # build the knowledge base
python -m citelocal_agent.ask "What is scaled dot-product attention?"
python -m citelocal_agent.chat                                    # multi-turn REPL
python -m citelocal_agent.web                                     # http://127.0.0.1:8000
```

## Testing

```bash
ruff check src tests scripts && mypy                       # lint + type check
python -m pytest tests/test_unit.py tests/test_retrieval.py -q   # offline, no key
python -m pytest tests/test_response.py tests/test_orchestrator.py tests/test_chat.py -q  # need a key
```

## Conventions

- The agent must answer **only** from retrieved content and always cite sources.
- `Answer` and `Question` are terminal tools — calling them ends the loop.
- Keep embeddings local; only the answer step may call an external LLM.
- Nothing is initialised at import time; models/tools are wired inside `build_agent`.
