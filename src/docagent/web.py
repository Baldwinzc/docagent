"""FastAPI web server for docagent: a JSON API + a static chat UI.

Run:
    python -m docagent.web              # then open http://127.0.0.1:8000
    # or: uvicorn docagent.web:app --reload

Endpoints:
    POST /api/ask      {question} -> {kind, intent, answer, question, citations, unsupported, trace}
    GET  /api/sources  -> {sources: [...]}
    GET  /             -> the chat UI (static/index.html)
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from docagent.agent import get_default_agent  # noqa: E402 (after load_dotenv)
from docagent.retriever import get_retriever  # noqa: E402
from docagent.utils import extract_outcome  # noqa: E402

app = FastAPI(
    title="docagent",
    description="Agentic RAG over local documents, with verified citations.",
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    kind: str
    intent: str
    answer: str
    question: str | None = None
    citations: list[str]
    unsupported: list[str]
    trace: list[dict]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = get_default_agent().invoke({"question_input": {"question": req.question}})
    o = extract_outcome(result)
    return AskResponse(
        kind=o["kind"],
        intent=o["intent"],
        answer=o["answer"],
        question=o["question"],
        citations=o["citations"],
        unsupported=o["unsupported"],
        trace=o["trace"],
    )


@app.get("/api/sources")
def sources() -> dict:
    return {"sources": get_retriever().list_sources()}


# Static chat UI (mounted last so /api/* routes take precedence).
_STATIC = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="ui")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
