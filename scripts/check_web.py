#!/usr/bin/env python
"""Smoke-test the web API and inspect citation grounding (needs an LLM API key).

    python scripts/check_web.py
"""

from fastapi.testclient import TestClient

from docagent.agent import get_default_agent
from docagent.utils import extract_outcome
from docagent.web import app


def main():
    c = TestClient(app)
    home = c.get("/")
    print(f"GET /            -> {home.status_code}  has-ui={'docagent' in home.text}")
    src = c.get("/api/sources").json().get("sources", [])
    print(f"GET /api/sources -> {len(src)} docs")

    # direct invoke so we can inspect grounding (retrieved locators vs citations)
    result = get_default_agent().invoke(
        {"question_input": {"question": "How do I declare an integer path parameter?"}},
        config={"recursion_limit": 12},
    )
    o = extract_outcome(result)
    print("retrieved   :", result.get("retrieved_locators"))
    print("intent      :", o["intent"])
    print("answer      :", (o["answer"] or "")[:140])
    print("citations   :", o["citations"])
    print("unsupported :", o["unsupported"])


if __name__ == "__main__":
    main()
