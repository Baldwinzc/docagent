"""Retrieval tests — no LLM API key required.

These exercise the full retrieval stack (hybrid dense+BM25 -> RRF ->
cross-encoder rerank -> relevance threshold) against the FastAPI corpus, so the
suite always has something that runs green offline (CI-friendly).
"""

from pathlib import Path

import pytest

from docagent.ingest import load_documents, chunk_documents
from docagent.vectorstore import get_vectorstore
from docagent.retriever import HybridRetriever

CORPUS = Path(__file__).parent.parent / "corpus" / "fastapi"


def test_load_corpus():
    docs = load_documents(CORPUS)
    assert len(docs) >= 10
    sources = {d.metadata["source"] for d in docs}
    assert "async.md" in sources
    # attribution files must be skipped, not ingested as content
    assert "LICENSE" not in sources and "SOURCE.md" not in sources


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    """Build a throwaway knowledge base from the corpus and return a retriever."""
    chroma_path = str(tmp_path_factory.mktemp("chroma"))
    docs = load_documents(CORPUS)
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=150)
    vs = get_vectorstore(persist_directory=chroma_path, collection_name="test_kb")
    vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return HybridRetriever(persist_directory=chroma_path, collection_name="test_kb")


def test_hybrid_retrieval_hits_relevant_doc(kb):
    hits = kb.search("how do I declare an integer path parameter", k=3)
    assert hits, "expected at least one hit"
    assert any("path-params" in h.source for h in hits)
    # every hit carries a precise line-range locator
    assert all(h.start_line is not None and h.end_line is not None for h in hits)


def test_hybrid_retrieval_error_handling_doc(kb):
    hits = kb.search("how to return an HTTP error to the client", k=3)
    assert any("handling-errors" in h.source for h in hits)


def test_threshold_filters_out_of_scope(kb):
    # nothing in the FastAPI docs answers this -> all candidates fall below
    # the relevance threshold, so the retriever returns nothing.
    hits = kb.search("what is the capital of France", k=3)
    assert hits == []
