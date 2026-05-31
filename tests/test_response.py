#!/usr/bin/env python
"""End-to-end agent tests (require an LLM API key; no LangSmith needed).

Two checks per in-scope question:
1. the agent makes the expected tool calls (search_docs -> Answer);
2. the final answer meets human-written criteria, graded by an LLM.

Run from the project root so the agent reads the ./chroma_db built by
`python -m docagent.ingest --path ./sample_docs`.
"""

import uuid

import pytest
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from docagent.agent import overall_workflow
from docagent.configuration import DEFAULT_LLM_MODEL
from docagent.utils import extract_tool_calls, format_messages_string
from docagent.eval.prompts import RESPONSE_CRITERIA_SYSTEM_PROMPT
from docagent.eval.qa_dataset import (
    qa_inputs,
    qa_names,
    response_criteria_list,
    intent_outputs,
    expected_tool_calls,
)


class CriteriaGrade(BaseModel):
    """LLM verdict on whether the answer meets the criteria."""

    grade: bool = Field(description="Does the response meet the criteria?")
    justification: str = Field(description="Justification with specific examples.")


criteria_eval_llm = init_chat_model(DEFAULT_LLM_MODEL).with_structured_output(
    CriteriaGrade
)


def _setup():
    compiled = overall_workflow.compile(checkpointer=MemorySaver())
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return compiled, thread_config


def _extract_values(state):
    return state.values if hasattr(state, "values") else state


def _in_scope_cases():
    cases = []
    for qa, name, criteria, intent, calls in zip(
        qa_inputs, qa_names, response_criteria_list, intent_outputs, expected_tool_calls
    ):
        if intent == "in_scope":
            cases.append((qa, name, criteria, calls))
    return cases


@pytest.mark.parametrize(
    "qa_input,qa_name,criteria,expected_calls", _in_scope_cases()
)
def test_expected_tool_calls(qa_input, qa_name, criteria, expected_calls):
    """The agent should search the KB and then call Answer."""
    compiled, thread_config = _setup()
    compiled.invoke({"question_input": qa_input}, config=thread_config)

    values = _extract_values(compiled.get_state(thread_config))
    extracted = extract_tool_calls(values["messages"])
    missing = [c for c in expected_calls if c.lower() not in extracted]
    assert not missing, f"[{qa_name}] missing tool calls {missing}; got {extracted}"


@pytest.mark.parametrize(
    "qa_input,qa_name,criteria,expected_calls", _in_scope_cases()
)
def test_response_criteria(qa_input, qa_name, criteria, expected_calls):
    """The final answer should satisfy the human-written criteria."""
    compiled, thread_config = _setup()
    compiled.invoke({"question_input": qa_input}, config=thread_config)

    values = _extract_values(compiled.get_state(thread_config))
    transcript = format_messages_string(values["messages"])

    result = criteria_eval_llm.invoke(
        [
            {"role": "system", "content": RESPONSE_CRITERIA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Response criteria: {criteria}\n\n"
                f"Assistant's response:\n{transcript}\n\n"
                "Does the response meet the criteria?",
            },
        ]
    )
    assert result.grade, f"[{qa_name}] {result.justification}"
