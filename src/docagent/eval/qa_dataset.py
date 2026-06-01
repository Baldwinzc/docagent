"""QA evaluation dataset over the demo arXiv papers.

Reproducible: download the same papers first, then run the eval:
    python scripts/fetch_arxiv.py --demo
    python -m docagent.ingest --path ./papers --reset
    python -m docagent.eval.run_eval

The set covers single-paper facts, a multi-hop question spanning two papers, an
out-of-scope question (router should decline), and an in-domain-sounding but
unanswerable question (the papers don't cover it, so the agent must say so).
"""

QA_CASES = [
    {
        "question": "What problem with recurrent models does the Transformer's attention mechanism address?",
        "intent": "in_scope",
        "expected_sources": ["attention-is-all-you-need.pdf"],
        "criteria": "The Transformer replaces recurrence with attention, enabling parallelization and better long-range dependencies.",
    },
    {
        "question": "What is scaled dot-product attention and why is the 1/sqrt(d_k) scaling used?",
        "intent": "in_scope",
        "expected_sources": ["attention-is-all-you-need.pdf"],
        "criteria": "Attention = softmax(QKᵀ/sqrt(d_k))V; the scaling keeps dot products from getting large and pushing softmax into small-gradient regions.",
    },
    {
        "question": "How does retrieval-augmented generation combine a retriever with a generator?",
        "intent": "in_scope",
        "expected_sources": ["retrieval-augmented-generation.pdf"],
        "criteria": "A retriever fetches relevant passages from an external index; the generator conditions on them (marginalizing over retrieved documents).",
    },
    {
        "question": "What pre-training objectives does BERT use?",
        "intent": "in_scope",
        "expected_sources": ["bert.pdf"],
        "criteria": "Masked language modeling (masked tokens) and next-sentence prediction.",
    },
    {
        "question": "Does RAG rely on the model's parametric memory or on a non-parametric memory?",
        "intent": "in_scope",
        "expected_sources": ["retrieval-augmented-generation.pdf"],
        "criteria": "A non-parametric memory — an external retrievable index — combined with the parametric model.",
    },
    {
        "question": "How is BERT related to the Transformer architecture?",
        "intent": "in_scope",
        "expected_sources": ["bert.pdf", "attention-is-all-you-need.pdf"],
        "criteria": "BERT is a multi-layer bidirectional Transformer encoder, built on the Transformer architecture.",
    },
    {
        "question": "What is the capital of France?",
        "intent": "out_of_scope",
        "expected_sources": [],
        "criteria": "Declines / states the question is outside the scope of the papers; does not answer 'Paris'.",
    },
    {
        "question": "How do I deploy a FastAPI application to production with Docker?",
        "intent": "no_answer",
        "expected_sources": [],
        "criteria": "Honestly states the papers do not cover this; does NOT fabricate deployment instructions.",
    },
]

# --- Derived views (used by tests/test_response.py) ---
qa_inputs = [{"question": c["question"]} for c in QA_CASES]
qa_names = [
    "transformer_vs_rnn", "scaled_attention", "rag_retriever_generator",
    "bert_pretraining", "rag_nonparametric", "bert_is_transformer",
    "out_of_scope_geo", "no_answer_deploy",
]
intent_outputs = [c["intent"] for c in QA_CASES]
response_criteria_list = [c["criteria"] for c in QA_CASES]
expected_tool_calls = [
    ["search_docs", "Answer"] if c["intent"] == "in_scope" else []
    for c in QA_CASES
]
