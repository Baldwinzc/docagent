"""State and structured-output schemas for the document QA agent."""

import operator
from typing import Annotated

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict


class IntentSchema(BaseModel):
    """Decide whether a question can be answered from the knowledge base."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["in_scope", "out_of_scope"] = Field(
        description="'in_scope' if the question is about the knowledge base contents "
        "and should be answered by retrieving documents; 'out_of_scope' for "
        "chit-chat or questions clearly unrelated to the documents.",
    )


class StateInput(TypedDict):
    """The external input to the graph."""

    question_input: dict  # e.g. {"question": "How do I declare a path param?"}


class State(MessagesState):
    """Graph state. ``messages`` is inherited from MessagesState.

    - ``trace`` accumulates observability events (one per retrieval / decision).
    - ``retrieved_locators`` accumulates every locator the agent actually
      retrieved this run, so the final answer's citations can be *verified*
      against what was really seen (not just trusted because the LLM emitted them).

    Both use the ``operator.add`` reducer so the whole run is inspectable.
    """

    question_input: dict
    classification_decision: Literal["in_scope", "out_of_scope"]
    trace: Annotated[list, operator.add]
    retrieved_locators: Annotated[list, operator.add]
