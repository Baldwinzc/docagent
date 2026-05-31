"""State and structured-output schemas for the document QA agent."""

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal
from langgraph.graph import MessagesState


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

    # e.g. {"question": "How do I configure X?"}
    question_input: dict


class State(MessagesState):
    """Graph state. ``messages`` is inherited from MessagesState."""

    question_input: dict
    classification_decision: Literal["in_scope", "out_of_scope"]
