"""QA evaluation dataset over the FastAPI documentation corpus.

Each case carries the expected intent, the expected source document(s) (for
retrieval-recall scoring), and a grading criterion (for LLM-judged answer
correctness). The set deliberately includes:
  - single-doc factual questions,
  - a multi-hop question spanning two docs,
  - an out-of-scope question (should be declined by the router),
  - an in-domain-sounding but unanswerable question (the docs don't cover it,
    so the agent must honestly say so rather than fabricate).

Ingest the corpus first:  python -m docagent.ingest --path ./corpus/fastapi --reset
"""

QA_CASES = [
    {
        "question": "How do I declare a path parameter that must be an integer, and what happens if the client sends a non-integer?",
        "intent": "in_scope",
        "expected_sources": ["tutorial-path-params.md"],
        "criteria": "Explains declaring the path param with a type like `item_id: int`, and that a non-integer value produces a validation error.",
    },
    {
        "question": "What is the difference between path parameters and query parameters?",
        "intent": "in_scope",
        "expected_sources": ["tutorial-query-params.md"],
        "criteria": "Path parameters are part of the URL path; query parameters come after `?` as key-value pairs and can be optional / have defaults.",
    },
    {
        "question": "How do I return a specific HTTP error response to the client?",
        "intent": "in_scope",
        "expected_sources": ["tutorial-handling-errors.md"],
        "criteria": "Mentions raising `HTTPException` with a status_code and detail.",
    },
    {
        "question": "How do I declare and receive a JSON request body?",
        "intent": "in_scope",
        "expected_sources": ["tutorial-body.md"],
        "criteria": "Declare a Pydantic `BaseModel` and use it as a function parameter type to receive the body.",
    },
    {
        "question": "When should I use async def versus a normal def for a path operation function?",
        "intent": "in_scope",
        "expected_sources": ["async.md"],
        "criteria": "Use `async def` when using awaitable libraries; plain `def` functions are run by FastAPI in an external threadpool.",
    },
    {
        "question": "What is dependency injection used for in FastAPI?",
        "intent": "in_scope",
        "expected_sources": ["tutorial-dependencies-index.md"],
        "criteria": "Lets you declare shared dependencies via `Depends` to reuse logic (auth, db sessions, common params).",
    },
    {
        "question": "How can I add minimum and maximum length validation to a string query parameter?",
        "intent": "in_scope",
        "expected_sources": ["tutorial-query-params-str-validations.md"],
        "criteria": "Use `Query` (typically with `Annotated`) and set `min_length` / `max_length`.",
    },
    {
        "question": "How do I declare an integer path parameter and also give a query parameter a default value?",
        "intent": "in_scope",
        "expected_sources": ["tutorial-path-params.md", "tutorial-query-params.md"],
        "criteria": "Covers both: typing the path param as `int`, and giving a query parameter a default value in the function signature.",
    },
    {
        "question": "What is the capital of France?",
        "intent": "out_of_scope",
        "expected_sources": [],
        "criteria": "Declines / states the question is outside the scope of the documents; does not answer 'Paris'.",
    },
    {
        "question": "How do I train a convolutional neural network in PyTorch?",
        "intent": "no_answer",
        "expected_sources": [],
        "criteria": "Honestly states the documents do not cover this topic; does NOT fabricate PyTorch instructions.",
    },
]

# --- Derived views (kept for the pytest suite in tests/test_response.py) ---
qa_inputs = [{"question": c["question"]} for c in QA_CASES]
qa_names = [
    "path_param_int", "path_vs_query", "http_error", "request_body",
    "async_vs_def", "dependency_injection", "str_validation",
    "multi_hop_path_query", "out_of_scope_geo", "no_answer_pytorch",
]
intent_outputs = [c["intent"] for c in QA_CASES]
response_criteria_list = [c["criteria"] for c in QA_CASES]
expected_tool_calls = [
    ["search_docs", "Answer"] if c["intent"] == "in_scope" else []
    for c in QA_CASES
]
