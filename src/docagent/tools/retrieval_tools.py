"""Retrieval tools for the document QA agent.

The agent drives an *agentic RAG* loop with these tools: it searches, inspects
results, optionally re-searches with a better query, and finally calls
``Answer`` — which forces it to attach citations, so no claim ships ungrounded.
"""

from typing import List

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from docagent.configuration import DEFAULT_TOP_K
from docagent.vectorstore import get_vectorstore


def _format_source(metadata: dict) -> str:
    """Build a short, human-readable source label from chunk metadata."""
    src = metadata.get("source", "unknown")
    page = metadata.get("page")
    return f"{src}#p{page}" if page is not None else str(src)


@tool
def search_docs(query: str, k: int = DEFAULT_TOP_K) -> str:
    """Search the knowledge base; return the top-k relevant chunks with sources."""
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    if not docs:
        return "No matching chunks found in the knowledge base."
    blocks = []
    for i, d in enumerate(docs, 1):
        label = _format_source(d.metadata or {})
        blocks.append(f"[{i}] source: {label}\n{d.page_content.strip()}")
    return "\n\n".join(blocks)


@tool
def list_sources() -> str:
    """List the distinct documents currently in the knowledge base."""
    vs = get_vectorstore()
    data = vs.get()  # all stored items; fine for a local KB
    metadatas = data.get("metadatas", []) or []
    sources = sorted({(m or {}).get("source", "unknown") for m in metadatas})
    if not sources:
        return "The knowledge base is empty. Run the ingest script first."
    return "Documents in the knowledge base:\n" + "\n".join(f"- {s}" for s in sources)


@tool
class Answer(BaseModel):
    """Provide the final, grounded answer with citations. Ends the task."""

    answer: str = Field(description="The final answer, grounded in retrieved chunks.")
    citations: List[str] = Field(
        description="Source labels (file name / page) actually used for the answer."
    )


@tool
class Question(BaseModel):
    """Ask the user a clarifying question when the request is too ambiguous."""

    content: str = Field(description="The clarifying question to ask the user.")
