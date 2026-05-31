"""Shared embedding + Chroma vector-store backend.

Both the ingestion script (``docagent.ingest``) and the retrieval tools
(``docagent.tools.retrieval_tools``) import from here, so they always read and
write the *same* collection with the *same* embedding model.

Embeddings run locally via sentence-transformers (no API key required), which
keeps the retrieval half of the project runnable for anyone who clones it.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from docagent.configuration import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """Return a cached local embedding function (downloads the model once)."""
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=4)
def get_vectorstore(
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """Return a cached, persistent Chroma handle for the given collection."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(embedding_model),
        persist_directory=persist_directory,
    )
