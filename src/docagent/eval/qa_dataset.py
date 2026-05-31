"""Small QA evaluation dataset over the bundled ``sample_docs`` knowledge base.

Each in-scope question targets a fact that actually appears in sample_docs/*.md,
so the agent must retrieve before it can answer correctly. One out-of-scope
question checks that the intent router declines politely.
"""

# Inputs to the graph: {"question_input": qa_inputs[i]}
qa_inputs = [
    {"question": "What vector store does docagent use to store chunks?"},
    {"question": "Which file formats can the ingest script read?"},
    {"question": "Do I need an API key to compute embeddings?"},
    {"question": "What is the capital of France?"},  # out of scope
]

qa_names = [
    "vector_store",
    "file_formats",
    "embeddings_api_key",
    "out_of_scope_geography",
]

# Expected intent-router decision per question
intent_outputs = ["in_scope", "in_scope", "in_scope", "out_of_scope"]

# For in-scope questions the agent should search and then answer
expected_tool_calls = [
    ["search_docs", "Answer"],
    ["search_docs", "Answer"],
    ["search_docs", "Answer"],
    [],  # out of scope: declined, no tool calls
]

# Grading criteria for the final answer (in-scope questions only)
response_criteria_list = [
    "States that docagent uses Chroma as its vector store.",
    "Mentions Markdown, plain text, and PDF as supported file formats.",
    "States that embeddings run locally and require no API key.",
    "",  # out of scope
]
