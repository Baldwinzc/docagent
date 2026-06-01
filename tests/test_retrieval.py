"""Retrieval tests — no LLM API key required.

These exercise the full retrieval stack (hybrid dense+BM25 -> RRF ->
cross-encoder rerank -> relevance threshold) against the bundled `sample_notes`,
so the suite always runs green offline / in CI without downloading any papers.
"""

from pathlib import Path

import pytest

from docagent.ingest import chunk_documents, load_documents
from docagent.retriever import HybridRetriever
from docagent.vectorstore import get_vectorstore

NOTES = Path(__file__).parent.parent / "sample_notes"


def test_load_notes():
    docs = load_documents(NOTES)
    assert len(docs) >= 3
    sources = {d.metadata["source"] for d in docs}
    assert "transformers-and-attention.md" in sources


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    """Build a throwaway knowledge base from sample_notes and return a retriever."""
    chroma_path = str(tmp_path_factory.mktemp("chroma"))
    docs = load_documents(NOTES)
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=120)
    vs = get_vectorstore(persist_directory=chroma_path, collection_name="test_kb")
    vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return HybridRetriever(persist_directory=chroma_path, collection_name="test_kb")


def test_hybrid_retrieval_hits_relevant_note(kb):
    hits = kb.search("what is scaled dot-product attention", k=3)
    assert hits, "expected at least one hit"
    assert any("attention" in h.source for h in hits)
    assert all(h.start_line is not None and h.end_line is not None for h in hits)


def test_hybrid_retrieval_bm25_keyword(kb):
    # BM25 should help match the exact term "BM25" even if dense misses it
    hits = kb.search("BM25 sparse retrieval", k=3)
    assert any("retrieval" in h.source for h in hits)


def test_threshold_filters_out_of_scope(kb):
    hits = kb.search("what is the capital of France", k=3)
    assert hits == []
