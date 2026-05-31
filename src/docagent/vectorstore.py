"""Shared embedding + Chroma vector-store backend.

Ingestion (``docagent.ingest``) and retrieval (``docagent.retriever``) import
from here, so they always read/write the same collection with the same local
embedding model (no API key required).
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
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


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
