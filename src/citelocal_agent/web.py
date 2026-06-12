"""FastAPI web server for citelocal_agent: a JSON API + a static chat UI.

Run:
    python -m citelocal_agent.web              # then open http://127.0.0.1:8000
    # or: uvicorn citelocal_agent.web:app --reload

Endpoints:
    GET  /health           -> {status} (liveness; no models loaded)
    POST /api/ask          {question, session_id?, collection?} -> {kind, intent, answer, ...}
    POST /api/ask/stream   same body -> Server-Sent Events: per-step + a final outcome
    GET  /api/sources      {collection?} -> {sources: [...]}
    GET  /                 -> the chat UI (static/index.html)

Hardening (all opt-in via env):
    DOCAGENT_API_KEY   when set, requests must send header `X-API-Key: <key>`
    RATE_LIMIT_REQUESTS / RATE_LIMIT_WINDOW   per-client fixed-window limit
Multi-tenant: pass ``collection`` to serve several knowledge bases from one server;
pass a stable ``session_id`` to hold a multi-turn conversation (else stateless).
"""

import json
import logging
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from citelocal_agent.agent import (  # noqa: E402 (after load_dotenv)
    build_agent,
    get_chat_agent,
)
from citelocal_agent.configuration import Configuration  # noqa: E402
from citelocal_agent.logging_config import configure_logging  # noqa: E402
from citelocal_agent.retriever import get_retriever  # noqa: E402
from citelocal_agent.security import RateLimiter, api_key_ok  # noqa: E402
from citelocal_agent.utils import extract_outcome  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="citelocal-agent",
    description="Agentic RAG over local documents, with verified citations.",
)

_rate_limiter = RateLimiter(
    max_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "60")),
    window_seconds=float(os.environ.get("RATE_LIMIT_WINDOW", "60")),
)


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if not api_key_ok(x_api_key):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def rate_limit(request: Request):
    client = request.client.host if request.client else "anon"
    if not _rate_limiter.allow(client, time.monotonic()):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


GUARDS = [Depends(require_api_key), Depends(rate_limit)]


@lru_cache(maxsize=8)
def _agent_for(collection: str | None):
    """Cached multi-turn agent per collection (one server, many knowledge bases)."""
    if not collection:
        return get_chat_agent()
    from langgraph.checkpoint.memory import InMemorySaver

    cfg = Configuration.from_runnable_config()
    cfg.collection_name = collection
    return build_agent(cfg, checkpointer=InMemorySaver())


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None   # supply to keep a multi-turn conversation
    collection: str | None = None   # supply to target a non-default knowledge base


class AskResponse(BaseModel):
    kind: str
    intent: str
    answer: str
    question: str | None = None
    citations: list[str]
    unsupported: list[str]
    trace: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse, dependencies=GUARDS)
def ask(req: AskRequest) -> AskResponse:
    thread_id = req.session_id or uuid.uuid4().hex
    result = _agent_for(req.collection).invoke(
        {"question_input": {"question": req.question}},
        config={"configurable": {"thread_id": thread_id}},
    )
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


@app.post("/api/ask/stream", dependencies=GUARDS)
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream progress as Server-Sent Events: one ``step`` event per graph node as
    it runs, then a ``final`` event with the verified outcome."""
    agent = _agent_for(req.collection)
    run_config = {"configurable": {"thread_id": req.session_id or uuid.uuid4().hex}}

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def gen():
        try:
            for update in agent.stream(
                {"question_input": {"question": req.question}},
                config=run_config,
                stream_mode="updates",
            ):
                for node in update:
                    yield sse({"event": "step", "node": node})
            o = extract_outcome(agent.get_state(run_config).values)
            yield sse(
                {
                    "event": "final",
                    "kind": o["kind"],
                    "intent": o["intent"],
                    "answer": o["answer"],
                    "question": o["question"],
                    "citations": o["citations"],
                    "unsupported": o["unsupported"],
                }
            )
        except Exception as e:  # noqa: BLE001 — surface as a stream error, don't 500
            logger.exception("stream failed")
            yield sse({"event": "error", "detail": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/sources", dependencies=GUARDS)
def sources(collection: str | None = None) -> dict:
    cfg = Configuration.from_runnable_config()
    retriever = get_retriever(cfg.chroma_path, collection or cfg.collection_name)
    return {"sources": retriever.list_sources()}


# Static chat UI (mounted last so /api/* routes take precedence).
_STATIC = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="ui")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
