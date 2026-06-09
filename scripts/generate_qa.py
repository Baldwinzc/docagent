#!/usr/bin/env python
"""Auto-generate QA evaluation cases from the ingested knowledge base.

This is the *generation* half of the eval workflow; the human-curation half is
editing the rows it emits into ``src/docagent/eval/data/qa_cases.jsonl``. The
generator **never** writes that curated file — it writes raw candidates to
``generated_raw.jsonl`` (gitignored, regenerable).

It draws real chunks from the same Chroma collection the agent answers from
(via ``get_vectorstore``), so every generated question has known provenance
(``source_chunk_ids``) and a known gold source. Categories:

    single_paper / numeric / definitional  - one chunk -> one question
    multi_hop                               - two chunks from two files -> a question needing both
    out_of_scope                            - no chunk; clearly-unrelated questions
    no_answer                               - in-domain-sounding questions, then
                                              VERIFIED against the real retriever
                                              to confirm the corpus does not cover them

Usage:
    python -m docagent.ingest --path ./papers --reset
    python scripts/generate_qa.py --n-per-category 25
    # then hand-curate generated_raw.jsonl into qa_cases.jsonl

Requires an LLM API key (uses the answer model); embeddings/retrieval run locally.
"""

import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from docagent.configuration import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_LLM_MODEL,
    DEFAULT_SCORE_THRESHOLD,
)
from docagent.retriever import get_retriever
from docagent.vectorstore import get_vectorstore

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "src" / "docagent" / "eval" / "data" / "generated_raw.jsonl"

# in_scope categories drawn from chunk(s); the other two are special-cased below.
CHUNK_CATEGORIES = ("single_paper", "numeric", "definitional", "multi_hop")
ALL_CATEGORIES = CHUNK_CATEGORIES + ("out_of_scope", "no_answer")

_CATEGORY_GUIDANCE = {
    "single_paper": "a factual question answerable from this single passage",
    "numeric": "a question about a specific number, formula, or quantity stated in this passage",
    "definitional": "a 'what is X' / 'define X' question this passage answers",
    "multi_hop": "a question that can ONLY be answered by combining BOTH passages "
    "(it must require a fact from each, not just one)",
}


class GeneratedQA(BaseModel):
    """A single auto-generated QA case (question + gold criterion)."""

    question: str = Field(description="A natural question a reader might ask.")
    criteria: str = Field(
        description="A concise statement of what a correct answer must contain, "
        "grounded strictly in the provided passage(s)."
    )


class CandidateQuestions(BaseModel):
    """A batch of in-domain-sounding questions the corpus likely does NOT cover."""

    questions: list[str] = Field(
        description="Questions that sound like they belong to the domain but are "
        "probably not answered by the documents."
    )


def _load_chunks(chroma_path: str, collection: str) -> list[dict]:
    """Return [{chunk_id, source, text}] for every chunk in the collection."""
    vs = get_vectorstore(persist_directory=chroma_path, collection_name=collection)
    data = vs.get()
    out = []
    for cid, text, meta in zip(
        data.get("ids", []), data.get("documents", []), data.get("metadatas", [])
    ):
        meta = meta or {}
        out.append(
            {
                "chunk_id": meta.get("chunk_id", cid),
                "source": meta.get("source", "unknown"),
                "text": text or "",
            }
        )
    return out


def _basename(source: str) -> str:
    return source.rsplit("/", 1)[-1]


def _row(case_id, question, intent, category, sources, criteria, chunk_ids, split):
    """Build a row matching the qa_cases.jsonl schema (curated=False)."""
    return {
        "id": case_id,
        "question": question.strip(),
        "intent": intent,
        "category": category,
        "expected_sources": sorted({_basename(s) for s in sources}),
        "criteria": criteria.strip(),
        "source_chunk_ids": chunk_ids,
        "curated": False,
        "split": split,
    }


def _gen_chunk_case(llm, category: str, chunks: list[dict]) -> GeneratedQA:
    """Prompt the LLM for one question grounded in the given chunk(s)."""
    passages = "\n\n".join(
        f"[Passage {i + 1}] (source: {c['source']})\n{c['text'].strip()}"
        for i, c in enumerate(chunks)
    )
    sys = (
        "You write evaluation questions for a document-QA system. Given one or "
        "more passages, produce ONE question and a concise gold criterion stating "
        "what a correct answer must contain. Ground both strictly in the passages; "
        "do not rely on outside knowledge. Make the question self-contained "
        "(do not say 'this passage')."
    )
    user = (
        f"Category: {category} — {_CATEGORY_GUIDANCE[category]}.\n\n"
        f"{passages}\n\nWrite the question and criterion."
    )
    return llm.invoke([{"role": "system", "content": sys}, {"role": "user", "content": user}])


def _gen_out_of_scope(llm, n: int, sources: list[str]) -> list[str]:
    topics = ", ".join(sorted({_basename(s) for s in sources})[:10])
    sys = (
        "You generate questions that are CLEARLY UNRELATED to a document collection "
        "(everyday trivia, greetings, general knowledge a technical corpus would "
        "never contain). They must be obviously off-topic."
    )
    user = (
        f"The collection is about: {topics}.\n"
        f"Produce {n} clearly off-topic questions."
    )
    batch = llm.with_structured_output(CandidateQuestions).invoke(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    )
    return batch.questions[:n]


def _gen_no_answer_candidates(llm, n: int, sources: list[str]) -> list[str]:
    topics = ", ".join(sorted({_basename(s) for s in sources})[:10])
    sys = (
        "You generate questions that SOUND like they belong to a technical domain "
        "but are probably NOT answered by these particular documents (adjacent "
        "topics, deployment/ops details, specific numbers unlikely to be stated)."
    )
    user = (
        f"The documents cover: {topics}.\n"
        f"Produce {2 * n} in-domain-sounding questions the documents likely do NOT answer."
    )
    batch = llm.with_structured_output(CandidateQuestions).invoke(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    )
    return batch.questions


def main():
    p = argparse.ArgumentParser(description="Auto-generate QA eval cases from the KB.")
    p.add_argument("--n-per-category", type=int, default=20)
    p.add_argument(
        "--categories", nargs="*", default=list(ALL_CATEGORIES),
        choices=ALL_CATEGORIES, help="Which categories to generate.",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--split", default="full_corpus", choices=["full_corpus", "offline_sample"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    chunks = _load_chunks(args.chroma_path, args.collection)
    if not chunks:
        raise SystemExit("Knowledge base is empty — ingest documents first.")
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(c)
    sources = list(by_source)
    print(f"Loaded {len(chunks)} chunks from {len(sources)} sources.")

    qa_llm = init_chat_model(DEFAULT_LLM_MODEL, temperature=0.3).with_structured_output(
        GeneratedQA
    )
    plain_llm = init_chat_model(DEFAULT_LLM_MODEL, temperature=0.3)

    rows: list[dict] = []
    n = args.n_per_category

    for category in args.categories:
        if category in ("single_paper", "numeric", "definitional"):
            pool = [c for c in chunks if len(c["text"]) > 200] or chunks
            picks = rng.sample(pool, min(n, len(pool)))
            for i, c in enumerate(picks):
                try:
                    g = _gen_chunk_case(qa_llm, category, [c])
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {category} #{i}: {e}")
                    continue
                rows.append(_row(
                    f"gen_{category}_{i}", g.question, "in_scope", category,
                    [c["source"]], g.criteria, [c["chunk_id"]], args.split,
                ))

        elif category == "multi_hop":
            if len(sources) < 2:
                print("  skip multi_hop: need >=2 sources")
                continue
            for i in range(n):
                s1, s2 = rng.sample(sources, 2)
                c1, c2 = rng.choice(by_source[s1]), rng.choice(by_source[s2])
                try:
                    g = _gen_chunk_case(qa_llm, "multi_hop", [c1, c2])
                except Exception as e:  # noqa: BLE001
                    print(f"  skip multi_hop #{i}: {e}")
                    continue
                rows.append(_row(
                    f"gen_multi_hop_{i}", g.question, "in_scope", "multi_hop",
                    [c1["source"], c2["source"]], g.criteria,
                    [c1["chunk_id"], c2["chunk_id"]], args.split,
                ))

        elif category == "out_of_scope":
            for i, q in enumerate(_gen_out_of_scope(plain_llm, n, sources)):
                rows.append(_row(
                    f"gen_out_of_scope_{i}", q, "out_of_scope", "out_of_scope",
                    [], "Declines; the question is unrelated to the documents.", [], args.split,
                ))

        elif category == "no_answer":
            # Generate candidates, then keep only those the real retriever can't cover.
            retriever = get_retriever(args.chroma_path, args.collection)
            kept = 0
            for q in _gen_no_answer_candidates(plain_llm, n, sources):
                if kept >= n:
                    break
                hits = retriever.search(q, score_threshold=DEFAULT_SCORE_THRESHOLD)
                if hits:  # corpus actually covers it -> not a no_answer case
                    continue
                rows.append(_row(
                    f"gen_no_answer_{kept}", q, "no_answer", "no_answer",
                    [], "Honestly states the documents do not cover this; does not fabricate.",
                    [], args.split,
                ))
                kept += 1
            print(f"  no_answer: kept {kept} verified-unanswerable questions.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} candidate cases to {out_path}.")
    print("Next: review/edit them, set curated=true, and merge into qa_cases.jsonl.")


if __name__ == "__main__":
    main()
