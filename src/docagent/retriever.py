"""Hybrid retriever — docagent's retrieval core.

Pipeline:  dense (vector) + sparse (BM25)  ->  Reciprocal-Rank-Fusion
           ->  cross-encoder rerank  ->  relevance-threshold filter.

This lifts results above naive top-k cosine similarity, and the threshold gives
the agent a principled way to say "not in the docs" (everything fell below it).
Each result carries precise provenance (file + line range, or PDF page) for
exact citations.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from docagent.configuration import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RRF_K,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
)
from docagent.vectorstore import get_vectorstore


@dataclass
class RetrievedChunk:
    """A retrieved chunk with provenance and a cross-encoder relevance score."""

    text: str
    source: str
    chunk_id: str
    score: float
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    page: Optional[int] = None

    @property
    def locator(self) -> str:
        """Precise, human-readable source locator used in citations."""
        if self.page is not None:
            return f"{self.source} (p.{self.page})"
        if self.start_line is not None:
            return f"{self.source}:L{self.start_line}-{self.end_line}"
        return self.source


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


@lru_cache(maxsize=2)
def _get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


class HybridRetriever:
    """Dense + BM25 retrieval, RRF fusion, cross-encoder rerank, threshold."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ):
        self.vs = get_vectorstore(
            persist_directory=persist_directory, collection_name=collection_name
        )
        data = self.vs.get()  # all stored items; fine for a local KB
        self.ids: List[str] = data.get("ids", []) or []
        self.docs: List[str] = data.get("documents", []) or []
        self.metas: List[dict] = data.get("metadatas", []) or []
        self._by_id = {
            cid: (text, meta or {})
            for cid, text, meta in zip(self.ids, self.docs, self.metas)
        }
        self._bm25 = BM25Okapi([_tokenize(d) for d in self.docs]) if self.docs else None
        self._reranker = _get_reranker(reranker_model)

    @property
    def is_empty(self) -> bool:
        return not self.docs

    def _dense_ids(self, query: str, candidate_k: int) -> List[str]:
        results = self.vs.similarity_search(query, k=candidate_k)
        return [
            (d.metadata or {}).get("chunk_id")
            for d in results
            if (d.metadata or {}).get("chunk_id") is not None
        ]

    def _bm25_ids(self, query: str, candidate_k: int) -> List[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked[:candidate_k]]

    @staticmethod
    def _rrf(rank_lists: List[List[str]], rrf_k: int) -> List[str]:
        fused: dict = {}
        for ids in rank_lists:
            for rank, cid in enumerate(ids):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused, key=lambda c: fused[c], reverse=True)

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> List[RetrievedChunk]:
        if self.is_empty:
            return []
        dense = self._dense_ids(query, candidate_k)
        sparse = self._bm25_ids(query, candidate_k)
        fused = [c for c in self._rrf([dense, sparse], rrf_k) if c in self._by_id]
        fused = fused[:candidate_k]
        if not fused:
            return []

        pairs = [(query, self._by_id[cid][0]) for cid in fused]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(fused, scores), key=lambda x: float(x[1]), reverse=True)

        out: List[RetrievedChunk] = []
        for cid, score in ranked:
            if float(score) < score_threshold:
                continue
            text, meta = self._by_id[cid]
            out.append(
                RetrievedChunk(
                    text=text,
                    source=meta.get("source", "unknown"),
                    chunk_id=cid,
                    score=float(score),
                    start_line=meta.get("start_line"),
                    end_line=meta.get("end_line"),
                    page=meta.get("page"),
                )
            )
            if len(out) >= k:
                break
        return out

    def list_sources(self) -> List[str]:
        return sorted({(m or {}).get("source", "unknown") for m in self.metas})


@lru_cache(maxsize=2)
def get_retriever(
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> HybridRetriever:
    """Cached retriever (loads chunks + BM25 + cross-encoder once)."""
    return HybridRetriever(
        persist_directory=persist_directory, collection_name=collection_name
    )
