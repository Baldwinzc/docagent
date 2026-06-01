"""Retrieval tools for the document QA agent.

`search_docs` runs the hybrid retriever (dense + BM25 -> RRF -> cross-encoder
rerank -> relevance threshold) and returns ranked chunks with **precise source
locators**, so the agent can cite exactly where each fact came from. The
terminal `Answer` tool forces citations, so no claim ships ungrounded.
"""

from typing import List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from docagent.configuration import DEFAULT_TOP_K
from docagent.retriever import get_retriever


@tool
def search_docs(query: str, k: int = DEFAULT_TOP_K) -> str:
    """Search the knowledge base; return reranked chunks with source locators.

    Each result is labelled with a precise locator (``file:Lstart-Lend`` or a
    PDF page) and a relevance score. Cite these locators in your final Answer.
    Call again with a reformulated query if the results are weak.
    """
    chunks = get_retriever().search(query, k=k)
    if not chunks:
        return (
            "No sufficiently relevant chunks found. The knowledge base likely "
            "does not cover this — reformulate the query, or if it still finds "
            "nothing, tell the user the answer is not in the documents."
        )
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[{i}] locator: {c.locator}  (relevance {c.score:.2f})\n{c.text.strip()}"
        )
    return "\n\n".join(blocks)


@tool
def list_sources() -> str:
    """List the distinct documents currently in the knowledge base."""
    sources = get_retriever().list_sources()
    if not sources:
        return "The knowledge base is empty. Run the ingest script first."
    return "Documents in the knowledge base:\n" + "\n".join(f"- {s}" for s in sources)


@tool
class Answer(BaseModel):
    """Provide the final, grounded answer with precise citations. Ends the task."""

    answer: str = Field(
        description="The final answer. Cite sources inline using their locators, "
        "e.g. '... runs it in a threadpool [async.md:L120-145].'"
    )
    citations: List[str] = Field(
        description="The exact source locators you relied on, e.g. "
        "['async.md:L120-145', 'tutorial-body.md:L5-30']."
    )


@tool
class Question(BaseModel):
    """Ask the user a clarifying question when the request is too ambiguous."""

    content: str = Field(description="The clarifying question to ask the user.")
