# docagent FAQ

## Do I need an API key?

The retrieval half (embeddings + vector search) runs fully locally and needs no
API key. The answer-generation LLM does need access to a model: by default it
uses OpenAI, but you can point it at any provider supported by
`init_chat_model`, including a local Ollama model.

## Which file formats are supported?

The ingest script reads Markdown (`.md`, `.markdown`), plain text (`.txt`),
reStructuredText (`.rst`), and PDF (`.pdf`) files.

## How do I switch the LLM?

Set the `LLM_MODEL` environment variable. For example, `openai:gpt-4.1` for
OpenAI or `ollama:llama3.1` for a local Ollama model.

## Where is my data stored?

All chunks and embeddings live in a local Chroma database directory (default
`./chroma_db`). Nothing about the document contents is sent anywhere except the
text you ask the answer LLM to reason over.

## How do I re-index after changing my documents?

Run the ingest script again with the `--reset` flag to clear the collection
before re-ingesting.
