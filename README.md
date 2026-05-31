# docagent — agentic RAG over your local documents

Ask natural-language questions over your own files (Markdown / text / PDF) and get
answers that **always cite their sources**. Built on [LangGraph](https://langchain-ai.github.io/langgraph/).

Unlike a plain single-shot RAG pipeline, docagent runs an **agentic retrieval
loop**: it decides how many times to search, reformulates weak queries, and only
answers once it has gathered enough evidence — through an `Answer` tool that
*forces* citations, so no claim ships ungrounded.

## Features

- 🔁 **Agentic retrieval** — the agent can call `search_docs` several times with
  reformulated queries before it commits to an answer.
- 📎 **Forced citations** — the final answer is produced by an `Answer` tool that
  requires a list of source citations.
- 🧭 **Intent routing** — an up-front router declines out-of-scope questions
  without wasting a retrieval or an answer.
- 🔒 **Local embeddings, no API key** — documents are embedded with
  sentence-transformers locally; only the answer step calls an LLM.
- 🧱 **Clean LangGraph architecture** — a small, readable two-layer state graph
  with tests, evaluation, and CI scaffolding.

## Architecture

```mermaid
flowchart LR
    START([START]) --> R[intent_router]
    R -- out_of_scope --> D([END · politely declined])
    R -- in_scope --> A

    subgraph A [response agent · RAG loop]
        direction LR
        L[llm_call] --> C{terminal tool?}
        C -- search_docs / list_sources --> E[environment]
        E --> L
        C -- Answer / Question --> X([END])
    end

    A --> OUT([answer + citations])
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

# 3. Build the knowledge base from the bundled samples
python -m docagent.ingest --path ./sample_docs

# 4. Ask away
python -m docagent.ask "What vector store does docagent use?"
python -m docagent.ask "What is the capital of France?"   # politely declined
```

Point `--path` at any folder of your own `.md` / `.txt` / `.rst` / `.pdf` files to
build a knowledge base over your own documents. Re-run with `--reset` to rebuild.

## How it works

1. **Intent router** classifies the question as `in_scope` or `out_of_scope`
   using an LLM with structured output. Out-of-scope questions end with a polite
   refusal.
2. **Response agent** (in-scope only) runs a tool-calling loop:
   - `search_docs` performs semantic search over the Chroma store;
   - the agent inspects results and may search again with a better query;
   - `Answer(answer, citations)` ends the loop with a grounded, cited answer.

## Project layout

```
src/docagent/
├── agent.py            # LangGraph: intent_router + response-agent RAG loop
├── ingest.py           # CLI: load docs → chunk → embed → Chroma
├── ask.py              # CLI: ask the knowledge base a question
├── vectorstore.py      # shared embeddings + Chroma backend
├── configuration.py    # env-overridable settings
├── prompts.py          # intent + agent prompts
├── schemas.py          # graph state + structured-output schemas
├── tools/
│   ├── base.py         # tool registry
│   └── retrieval_tools.py  # search_docs, list_sources, Answer, Question
└── eval/               # QA dataset + grading prompt
sample_docs/            # demo knowledge base
tests/                  # local retrieval tests + LLM end-to-end tests
```

## Testing

```bash
python tests/run_all_tests.py         # local retrieval tests (no API key)
python tests/run_all_tests.py --all   # + LLM end-to-end (needs API key)
```

## Configuration

All settings can be set in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Key for the default OpenAI answer model |
| `LLM_MODEL` | `openai:gpt-4.1` | Any `init_chat_model` id, e.g. `ollama:llama3.1` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHROMA_PATH` | `./chroma_db` | Vector store directory |
| `CHROMA_COLLECTION` | `docagent` | Collection name |
| `TOP_K` | `4` | Chunks retrieved per search |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Ingest chunking |

## Tech stack

LangGraph · LangChain · Chroma · sentence-transformers · pypdf

## License

MIT
