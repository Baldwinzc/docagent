# docagent — agentic RAG over your local documents

**English** | [中文](README.zh-CN.md)

Ask natural-language questions over your own files (Markdown / text / PDF) and get
answers that **cite their exact source location**. Built on
[LangGraph](https://langchain-ai.github.io/langgraph/).

docagent is not a single-shot RAG demo. It runs an **agentic retrieval loop** on
top of a real retrieval pipeline (hybrid dense + BM25, cross-encoder rerank,
relevance threshold), forces grounded citations down to the line range, exposes a
retrieval trace, and ships with a **quantitative evaluation** harness.

## Features

- 🔁 **Agentic retrieval** — the agent can search, inspect, reformulate, and
  search again before answering.
- 🧪 **Hybrid retrieval + rerank** — dense (bge embeddings) **+** BM25 fused with
  Reciprocal-Rank-Fusion, then re-ranked by a cross-encoder, then filtered by a
  relevance threshold (which is also how it honestly says "not in the docs").
- 📎 **Precise, forced citations** — answers cite exact locators like
  `tutorial-path-params.md:L65-91`, produced through an `Answer` tool that
  *requires* citations.
- 🧭 **Intent routing** — an up-front router declines out-of-scope questions.
- 🔭 **Observability** — every retrieval step is recorded in a `trace`
  (`--trace` on the CLI).
- 🛡️ **Robustness** — empty-KB guard, tool-failure capture, recursion limit.
- 📊 **Quantitative evaluation** — intent / recall / answer / citation / refusal
  metrics over a labelled QA set (see [Evaluation](#evaluation)).
- 🔒 **Local embeddings, no API key** for retrieval; only the answer LLM needs one.

## Architecture

```mermaid
flowchart TB
    START([START]) --> R[intent_router]
    R -- out_of_scope / empty KB --> D([END · declined])
    R -- in_scope --> L[llm_call]
    L --> C{terminal tool?}
    C -- search_docs --> E[environment]
    E --> L
    C -- Answer --> X([END · answer + citations])

    E -.calls.-> PIPE

    subgraph PIPE [hybrid retrieval pipeline]
      direction TB
      Q[query] --> DENSE[dense · bge]
      Q --> BM25[BM25]
      DENSE --> RRF[RRF fusion]
      BM25 --> RRF
      RRF --> RERANK[cross-encoder rerank]
      RERANK --> THRESH[relevance threshold]
      THRESH --> OUT2[top-k chunks + locators]
    end
```

## Quickstart

```bash
# 1. Environment (Python 3.11)
conda create -n docagent python=3.11 -c conda-forge
conda activate docagent
pip install -e .

# 2. Configure the answer LLM
cp .env.example .env          # then put your OPENAI_API_KEY in .env
#   (or set LLM_MODEL=ollama:llama3.1 to run fully local)

# 3. Build the knowledge base (bundled FastAPI docs, or point --path at your own)
python -m docagent.ingest --path ./corpus/fastapi --reset

# 4. Ask
python -m docagent.ask --trace "How do I declare an integer path parameter?"
```

## Example run

**In-scope question** — the agent searches, then answers with line-precise citations:

```console
$ python -m docagent.ask --trace "How do I declare a path parameter that must be an integer, and what does FastAPI do if the client sends a non-integer?"
🔎 Intent: IN_SCOPE — retrieving from knowledge base
=== trace ===
  1. search_docs  query='FastAPI path parameter integer non-integer validation'

=== Answer ===
Declare the path parameter with a Python type annotation, e.g. `item_id: int`.
FastAPI validates the value and returns a validation error if the client sends a
non-integer [tutorial-path-params.md:L65-91].

=== Citations ===
- tutorial-path-params.md:L65-91
- tutorial-path-params.md:L89-107
```

**Out-of-scope question** — the router declines without wasting a retrieval:

```console
$ python -m docagent.ask "What is the capital of France?"
🚫 Intent: OUT_OF_SCOPE — politely declining
This question is outside the scope of the local knowledge base, so I can't
answer it from the available documents.
```

You can inspect the retrieval stack directly (no API key) with the probe:

```bash
python scripts/check_retrieval.py
```

## Retrieval pipeline

`search_docs` does **not** do naive top-k cosine similarity. For each query:

1. **Dense** retrieval with `bge-small-en-v1.5` embeddings (top *candidate_k*).
2. **Sparse** retrieval with **BM25** (top *candidate_k*).
3. **RRF fusion** combines both rankings (robust to either signal being weak).
4. **Cross-encoder rerank** (`ms-marco-MiniLM-L-6-v2`) scores each candidate
   against the query.
5. **Relevance threshold** drops low-scoring chunks — so an out-of-domain query
   returns nothing, which is what lets the agent answer "not in the docs".

Each surviving chunk keeps precise provenance (`source:Lstart-Lend`, or a PDF
page) for citation.

## Evaluation

A labelled QA set (`src/docagent/eval/qa_dataset.py`) covers single-doc,
multi-hop, out-of-scope, and unanswerable questions. Run it with:

```bash
python -m docagent.eval.run_eval
```

Latest run over the bundled FastAPI corpus (206 chunks / 12 docs), answer LLM
`gpt-5.4-mini`:

| Metric | Result |
|---|---|
| Intent routing accuracy | **10/10 (100%)** |
| Retrieval recall (mean) | **0.94** |
| Answer correctness (LLM-judged) | **7/8 (88%)** |
| Citation grounding | **7/8 (88%)** |
| Refusal accuracy | **2/2 (100%)** |

The one miss is a **multi-hop** question that needs facts from two documents at
once; the agent retrieves one of them. Multi-hop synthesis (sub-query
decomposition) is the main area for future work.

## Project layout

```
src/docagent/
├── agent.py            # LangGraph: intent_router + response-agent loop + trace/guards
├── retriever.py        # hybrid retrieval: dense+BM25 -> RRF -> rerank -> threshold
├── ingest.py           # CLI: load -> chunk (+ line provenance) -> embed -> Chroma
├── ask.py              # CLI: ask the KB (--trace to see retrieval steps)
├── vectorstore.py      # shared embeddings + Chroma backend
├── configuration.py    # env-overridable settings
├── prompts.py          # intent + agent prompts
├── schemas.py          # graph state (+ trace channel) + structured-output schemas
├── tools/
│   ├── base.py         # tool registry
│   └── retrieval_tools.py  # search_docs, list_sources, Answer, Question
└── eval/
    ├── qa_dataset.py   # labelled QA cases
    └── run_eval.py     # metrics: intent / recall / answer / citation / refusal
corpus/fastapi/         # demo corpus (FastAPI docs subset, MIT — see SOURCE.md)
scripts/check_retrieval.py  # dev probe for the retrieval stack (no API key)
tests/                  # retrieval tests (no key) + LLM end-to-end tests
```

## Testing

```bash
python tests/run_all_tests.py          # retrieval tests only (no API key)
python tests/run_all_tests.py --all    # + LLM end-to-end (needs API key)
```

`tests/test_retrieval.py` exercises the hybrid retriever (hits, line-precise
locators, threshold) with **no API key** and runs in CI.

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Key for the answer model |
| `LLM_MODEL` | `openai:gpt-4.1` | Any `init_chat_model` id, e.g. `ollama:llama3.1` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local dense embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `TOP_K` / `CANDIDATE_K` | `4` / `20` | Final hits / per-retriever candidates |
| `SCORE_THRESHOLD` | `0.0` | Min rerank score to keep a chunk |
| `CHROMA_PATH` / `CHROMA_COLLECTION` | `./chroma_db` / `docagent` | Vector store |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Ingest chunking |

## Tech stack

LangGraph · LangChain · Chroma · sentence-transformers (bge) · rank-bm25 ·
cross-encoder · pypdf

## License

MIT (this project). The demo corpus under `corpus/fastapi/` is a subset of the
FastAPI documentation, redistributed under its MIT license — see
`corpus/fastapi/SOURCE.md`.
