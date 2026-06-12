#!/usr/bin/env python
"""Smoke-test the web API and inspect citation grounding (needs an LLM API key).

    python scripts/check_web.py
"""

from fastapi.testclient import TestClient

from citelocal_agent.agent import get_default_agent
from citelocal_agent.utils import extract_outcome
from citelocal_agent.web import app


def main():
    c = TestClient(app)
    home = c.get("/")
    print(f"GET /            -> {home.status_code}  has-ui={'citelocal_agent' in home.text}")
    src = c.get("/api/sources").json().get("sources", [])
    print(f"GET /api/sources -> {len(src)} docs")

    # direct invoke so we can inspect grounding (retrieved locators vs citations)
    result = get_default_agent().invoke(
        {"question_input": {"question": "How do I declare an integer path parameter?"}}
    )
    o = extract_outcome(result)
    print("retrieved   :", result.get("retrieved_locators"))
    print("intent      :", o["intent"])
    print("answer      :", (o["answer"] or "")[:140])
    print("citations   :", o["citations"])
    print("unsupported :", o["unsupported"])


if __name__ == "__main__":
    main()
