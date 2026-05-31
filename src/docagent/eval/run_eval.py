#!/usr/bin/env python
"""Quantitative evaluation of docagent over docagent.eval.qa_dataset.

Run from the project root, after ingesting the corpus:
    python -m docagent.ingest --path ./corpus/fastapi --reset
    python -m docagent.eval.run_eval

Metrics:
    intent    - router decision matches expected (in_scope vs out_of_scope)
    recall    - expected source doc(s) present in retriever top-k
    answer    - LLM judges the final answer meets the criterion (in_scope cases)
    citation  - citations point to an expected source (in_scope sourced cases)
    refusal   - out_of_scope / no_answer cases correctly declined
"""

import re

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from docagent.agent import docagent
from docagent.retriever import get_retriever
from docagent.configuration import DEFAULT_LLM_MODEL, DEFAULT_TOP_K
from docagent.eval.qa_dataset import QA_CASES, qa_names

load_dotenv()


class Grade(BaseModel):
    correct: bool = Field(description="Does the answer satisfy the criterion?")
    justification: str = Field(description="Brief justification.")


_judge = init_chat_model(DEFAULT_LLM_MODEL).with_structured_output(Grade)

JUDGE_SYS = (
    "You grade a document-QA assistant's answer against a single criterion. "
    "Return correct=true only if the answer satisfies the criterion. When the "
    "criterion requires declining, correct=true means the answer appropriately "
    "declines or says the documents don't cover it (and does not fabricate)."
)


def _source_of(locator: str) -> str:
    """'file.md:L1-29' or 'file.pdf (p.3)' -> 'file...'."""
    return re.split(r"[:(]", locator.strip())[0].strip()


def _run_agent(question: str):
    result = docagent.invoke(
        {"question_input": {"question": question}}, config={"recursion_limit": 12}
    )
    intent = result.get("classification_decision")
    answer, citations = "", []
    for msg in reversed(result.get("messages", [])):
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc["name"] == "Answer":
                answer = tc["args"].get("answer", "")
                citations = tc["args"].get("citations", []) or []
                break
        if answer:
            break
    if not answer and result.get("messages"):
        answer = str(result["messages"][-1].content)
    return intent, answer, citations


def _judge_answer(criteria: str, question: str, answer: str) -> bool:
    grade = _judge.invoke(
        [
            {"role": "system", "content": JUDGE_SYS},
            {
                "role": "user",
                "content": f"Question: {question}\n\nCriterion: {criteria}\n\nAnswer:\n{answer}",
            },
        ]
    )
    return bool(grade.correct)


def _mark(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "ok" if v else "X"
    return f"{v:.2f}"


def main():
    retriever = get_retriever()
    print(f"Corpus: {len(retriever.docs)} chunks / {len(retriever.list_sources())} docs\n")

    intent_c = intent_t = ans_c = ans_t = cite_c = cite_t = ref_c = ref_t = 0
    recalls = []

    header = f"{'case':<22}{'intent':<8}{'recall':<8}{'answer':<8}{'citation':<10}{'refusal':<8}"
    print(header)
    print("-" * len(header))

    for case, name in zip(QA_CASES, qa_names):
        q = case["question"]
        exp_sources = set(case["expected_sources"])
        exp_intent = case["intent"]
        exp_router = "out_of_scope" if exp_intent == "out_of_scope" else "in_scope"

        hits = retriever.search(q, k=DEFAULT_TOP_K)
        retrieved = {h.source for h in hits}
        recall = None
        if exp_sources:
            recall = len(exp_sources & retrieved) / len(exp_sources)
            recalls.append(recall)

        intent, answer, citations = _run_agent(q)
        cited = {_source_of(c) for c in citations}

        intent_ok = intent == exp_router
        intent_c += int(intent_ok)
        intent_t += 1

        answer_ok = _judge_answer(case["criteria"], q, answer)

        cite_ok = None
        if exp_intent == "in_scope" and exp_sources:
            cite_ok = len(cited & exp_sources) > 0
            cite_c += int(cite_ok)
            cite_t += 1

        refusal_ok = None
        if exp_intent in ("out_of_scope", "no_answer"):
            refusal_ok = answer_ok
            ref_c += int(refusal_ok)
            ref_t += 1
        else:
            ans_c += int(answer_ok)
            ans_t += 1

        ans_cell = answer_ok if exp_intent == "in_scope" else None
        print(
            f"{name:<22}{_mark(intent_ok):<8}{_mark(recall):<8}"
            f"{_mark(ans_cell):<8}{_mark(cite_ok):<10}{_mark(refusal_ok):<8}"
        )

    print("-" * len(header))

    def pct(c, t):
        return f"{c}/{t} ({100 * c / t:.0f}%)" if t else "n/a"

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print("\n=== SUMMARY ===")
    print(f"Intent routing accuracy : {pct(intent_c, intent_t)}")
    print(f"Retrieval recall (mean) : {mean_recall:.2f}")
    print(f"Answer correctness      : {pct(ans_c, ans_t)}")
    print(f"Citation grounding      : {pct(cite_c, cite_t)}")
    print(f"Refusal accuracy        : {pct(ref_c, ref_t)}")


if __name__ == "__main__":
    main()
