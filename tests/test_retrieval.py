"""Local retrieval tests — no LLM API key required.

These exercise the ingestion + vector-search half of the project end to end
against the bundled sample_docs, so the suite has something that always runs
green offline (CI-friendly).
"""

from pathlib import Path

import pytest

from docagent.ingest import load_documents
from docagent.vectorstore import get_vectorstore

SAMPLE_DOCS = Path(__file__).parent.parent / "sample_docs"


def test_load_sample_documents():
    docs = load_documents(SAMPLE_DOCS)
    assert len(docs) >= 3
    sources = {d.metadata["source"] for d in docs}
    assert "faq.md" in sources


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chroma_path = str(tmp_path_factory.mktemp("chroma"))
    docs = load_documents(SAMPLE_DOCS)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)
    vs = get_vectorstore(persist_directory=chroma_path, collection_name="test_kb")
    vs.add_documents(chunks)
    return vs


def test_search_finds_vector_store_fact(ingested):
    results = ingested.similarity_search("what vector store is used", k=3)
    joined = " ".join(r.page_content.lower() for r in results)
    assert "chroma" in joined


def test_search_finds_file_formats_fact(ingested):
    results = ingested.similarity_search("supported file formats", k=3)
    joined = " ".join(r.page_content.lower() for r in results)
    assert "pdf" in joined
