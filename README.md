# docagent — agentic RAG over your local documents

**English** | [中文](README.zh-CN.md)

Ask natural-language questions over a real documentation corpus and get answers
that **cite their exact source location** (file + line range, or PDF page). Built
on [LangGraph](https://langchain-ai.github.io/langgraph/), with a hybrid
retrieval pipeline, a quantitative eval harness, and a small web UI.

The bundled knowledge base is themed: **Modern Python Web Development** — the
FastAPI docs plus the Python typing/async PEPs they build on — across **Markdown,
reStructuredText, and PDF** (127 documents / ~1.25k chunks).

## Features

- 🔁 **Agentic retrieval** — search, inspect, reformulate, search again, then answer.
- 🧪 **Hybrid retrieval + rerank** — dense (bge) **+** BM25 fused with RRF, then a
  cross-encoder rerank, then a relevance threshold (also how it says "not in the docs").
- 🗂️ **Multi-format** — Markdown, reStructuredText, and PDF in one knowledge base,
  each citation carrying the right locator (`file.md:L10-30` or `file.pdf (p.3)`).
- 📎 **Verified citations** — the `Answer` tool *requires* citations, and each one
  is **checked against what was actually retrieved**; unsupported (hallucinated)
  locators are dropped rather than trusted.
- 🧭 **Intent routing**, 🔭 **retrieval trace** (`--trace`), 🛡️ **robustness**
  (empty-KB / tool-failure / recursion guards).
- 📊 **Quantitative evaluation** — intent / recall / answer / citation / refusal.
- 💬 **Web UI** — a small FastAPI + static chat front-end.
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
pip install -e .          # extras: ".[dev]" tests/lint · ".[cli]" langgraph dev · ".[corpus]" rebuild PDF

# 2. Configure the answer LLM
cp .env.example .env          # put OPENAI_API_KEY in .env (or LLM_MODEL=ollama:llama3.1)

# 3. Build the corpus (FastAPI docs + PEPs + a PDF) and index it
python scripts/build_corpus.py
python -m docagent.ingest --path ./corpus --reset

# 4a. Ask from the CLI
python -m docagent.ask --trace "How do I declare an integer path parameter?"

# 4b. …or launch the web UI
python -m docagent.web        # open http://127.0.0.1:8000
```

Point `ingest --path` at any folder of your own `.md` / `.rst` / `.txt` / `.pdf`
files to build a knowledge base over your own documents.

## Web UI

![docagent web UI](docs/ui-answer.png)

A small chat front-end (FastAPI backend + a static Tailwind page) shows the
answer, the intent badge, citation chips, and a collapsible retrieval trace:

```bash
python -m docagent.web   # http://127.0.0.1:8000
```

API: `POST /api/ask {question}` → `{kind, intent, answer, question, citations, unsupported, trace}`,
`GET /api/sources` → the document list.

## Example run

**Cross-format retrieval** (the probe, no API key — `python scripts/check_retrieval.py`):

```console
Q: What does PEP 484 specify about type hints?
   pep-0484.rst:L1-25                 score= 5.42  'PEP: 484 Title: Type Hints ...'
Q: What are protocols and structural subtyping?
   pep-0544-protocols.pdf (p.1)       score= 5.07  'PEP: 544 Title: Protocols: Structural subtyping ...'
Q: What is the capital of France?
   (no chunk passed the relevance threshold)
```

**In-scope question** (CLI):

```console
$ python -m docagent.ask --trace "How do I declare a path parameter that must be an integer, and what does FastAPI do if the client sends a non-integer?"
🔎 Intent: IN_SCOPE — retrieving from knowledge base
=== trace ===
  1. search_docs  query='FastAPI path parameter integer non-integer validation'

=== Answer ===
Declare the path parameter with a Python type annotation, e.g. `item_id: int`.
FastAPI validates it and returns a validation error for a non-integer
[tutorial-path-params.md:L65-91].

=== Citations ===
- tutorial-path-params.md:L65-91
```

## Corpus

Themed, multi-format, reproducible:

| Source | Format | Count | License |
|---|---|---|---|
| FastAPI docs (tutorial / advanced / how-to / deployment) | Markdown | 119 | MIT |
| Python PEPs (484, 492, 8, 257, 20, 585, 604) | reStructuredText | 7 | PSF |
| PEP 544 (Protocols) rendered to PDF | PDF | 1 | PSF |

Rebuild any time with `python scripts/build_corpus.py` (see `corpus/SOURCE.md`
for attribution).

## Retrieval pipeline

`search_docs` is not naive top-k cosine. Per query: **dense** (`bge-small-en-v1.5`)
+ **BM25** → **RRF fusion** → **cross-encoder rerank** (`ms-marco-MiniLM-L-6-v2`)
→ **relevance threshold**. Each surviving chunk keeps a precise locator for citation.

## Evaluation

A labelled QA set (`src/docagent/eval/qa_dataset.py`) covers single-doc,
multi-hop, out-of-scope, and unanswerable questions:

```bash
python -m docagent.eval.run_eval
```

Latest run over the bundled corpus (~1.25k chunks / 126 docs), answer LLM
`gpt-5.4-mini`:

| Metric | Result |
|---|---|
| Intent routing accuracy | **10/10 (100%)** |
| Retrieval recall (mean) | **0.94** |
| Answer correctness (LLM-judged) | **7–8/8 (88–100%, varies run-to-run)** |
| Citation grounding | **8/8 (100%)** |
| Hallucinated citations | **0** (every citation verified against retrieval) |
| Refusal accuracy | **2/2 (100%)** |

Hallucinated citations: **0** — every emitted citation was verified against what
was actually retrieved. Retrieval quality held as the corpus grew 6× (206 →
~1.25k chunks). This is a small, single-domain validation set: it exercises the
pipeline, it does **not** prove general-purpose scale (see [Limitations](#limitations)).

## Limitations

This is a portfolio-grade local-docs RAG, not a production system. Known limits:

- **Corpus scale** — the retriever loads all chunks and builds BM25 in memory at
  startup. Fine for a local KB (≤ ~10k chunks); for 10⁵–10⁶ chunks it needs a
  server-side sparse index and lazy loading.
- **Citation verification is source-level** — citations are checked against the
  retrieved locators (by file, and by exact locator when it matches), not yet by
  re-verifying that each sentence is entailed by the cited span.
- **Multi-hop** — questions that need two documents at once are the main accuracy
  gap; sub-query decomposition is future work.
- **Eval set is small & single-domain** (10 cases over Python-web docs) — enough
  to exercise the pipeline, not to claim broad generalisation.

## Project layout

```
src/docagent/
├── agent.py            # LangGraph: intent_router + response loop + trace/guards
├── retriever.py        # hybrid: dense+BM25 -> RRF -> rerank -> threshold
├── ingest.py           # load -> chunk (+line/page provenance) -> embed -> Chroma
├── ask.py / web.py     # CLI / FastAPI+static web UI
├── static/index.html   # chat front-end (Tailwind)
├── vectorstore.py, configuration.py, prompts.py, schemas.py
├── tools/              # search_docs, list_sources, Answer, Question
└── eval/               # qa_dataset.py + run_eval.py
corpus/{fastapi,peps,pdf}/   # themed multi-format demo corpus
scripts/                # build_corpus.py, check_retrieval.py, check_web.py
tests/                  # retrieval tests (no key) + LLM end-to-end tests
```

## Testing

```bash
python tests/run_all_tests.py          # retrieval tests only (no API key)
python tests/run_all_tests.py --all    # + LLM end-to-end (needs API key)
```

## Configuration

Key settings in `.env` (see `.env.example`): `OPENAI_API_KEY`, `LLM_MODEL`
(default `openai:gpt-4.1`; any `init_chat_model` id), `EMBEDDING_MODEL`
(`BAAI/bge-small-en-v1.5`), `RERANKER_MODEL`
(`cross-encoder/ms-marco-MiniLM-L-6-v2`), `TOP_K`/`CANDIDATE_K` (`4`/`20`),
`SCORE_THRESHOLD` (`0.0`), `CHROMA_PATH`/`CHROMA_COLLECTION`,
`CHUNK_SIZE`/`CHUNK_OVERLAP`.

## Tech stack

LangGraph · LangChain · Chroma · sentence-transformers (bge) · rank-bm25 ·
cross-encoder · pypdf · FastAPI · Tailwind

## License

MIT (this project). The demo corpus under `corpus/` redistributes FastAPI docs
(MIT) and Python PEPs (PSF) — see `corpus/SOURCE.md`.
