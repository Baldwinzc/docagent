"""QA evaluation dataset over the demo arXiv papers (and bundled sample notes).

The cases live in ``data/qa_cases.jsonl`` (one JSON object per line) so the set
can grow to hundreds of human-curated rows without bloating this module and so it
diffs cleanly in review. This module is just the **loader + derived views**; all
existing imports (``QA_CASES``, ``qa_names``, …) keep working.

Each row has:
    id              - unique, stable case id (also used as the case name)
    question        - the user question
    intent          - "in_scope" | "out_of_scope" | "no_answer"
    category        - "single_paper" | "multi_hop" | "out_of_scope"
                      | "no_answer" | "numeric" | "definitional"
    expected_sources- source basenames the answer should rest on ([] if none)
    criteria        - what a correct answer / refusal must satisfy (LLM-judged)
    source_chunk_ids- provenance of where a generated case was drawn from ([] if hand-written)
    curated         - True once a human has reviewed the row
    split           - "offline_sample" (answerable from bundled sample_notes/, used by CI)
                      | "full_corpus"  (needs the downloaded papers/, manual/nightly eval)

Reproducible full run (after downloading the papers):
    python scripts/fetch_arxiv.py --demo
    python -m docagent.ingest --path ./papers --reset
    python -m docagent.eval.run_eval

Required keys + the allowed enum values are validated on load so a malformed row
fails fast instead of silently skewing the metrics.
"""

import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "qa_cases.jsonl"

INTENTS = {"in_scope", "out_of_scope", "no_answer"}
CATEGORIES = {
    "single_paper",
    "multi_hop",
    "out_of_scope",
    "no_answer",
    "numeric",
    "definitional",
}
SPLITS = {"offline_sample", "full_corpus"}
_REQUIRED_KEYS = {
    "id",
    "question",
    "intent",
    "category",
    "expected_sources",
    "criteria",
    "split",
}


def _validate(case: dict, lineno: int) -> dict:
    missing = _REQUIRED_KEYS - case.keys()
    if missing:
        raise ValueError(f"qa_cases.jsonl line {lineno}: missing keys {sorted(missing)}")
    if case["intent"] not in INTENTS:
        raise ValueError(f"qa_cases.jsonl line {lineno}: bad intent {case['intent']!r}")
    if case["category"] not in CATEGORIES:
        raise ValueError(f"qa_cases.jsonl line {lineno}: bad category {case['category']!r}")
    if case["split"] not in SPLITS:
        raise ValueError(f"qa_cases.jsonl line {lineno}: bad split {case['split']!r}")
    # Tolerate older / generated rows that omit optional fields.
    case.setdefault("source_chunk_ids", [])
    case.setdefault("curated", False)
    return case


def load_qa_cases(
    split: str | None = None, path: str | Path = DATA_PATH
) -> list[dict[str, Any]]:
    """Load (and validate) the QA cases, optionally filtered to one ``split``."""
    cases: list[dict] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            case = _validate(json.loads(line), lineno)
            if case["id"] in seen_ids:
                raise ValueError(f"qa_cases.jsonl line {lineno}: duplicate id {case['id']!r}")
            seen_ids.add(case["id"])
            if split is None or case["split"] == split:
                cases.append(case)
    return cases


# --- Back-compat module global: the full set, in file order ---
QA_CASES = load_qa_cases()

# --- Derived views (all positionally aligned with QA_CASES; used by run_eval /
#     tests/test_response.py — keep them in lockstep) ---
qa_inputs = [{"question": c["question"]} for c in QA_CASES]
qa_names = [c["id"] for c in QA_CASES]
intent_outputs = [c["intent"] for c in QA_CASES]
response_criteria_list = [c["criteria"] for c in QA_CASES]
expected_tool_calls = [
    ["search_docs", "Answer"] if c["intent"] == "in_scope" else [] for c in QA_CASES
]
