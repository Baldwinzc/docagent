#!/usr/bin/env python
"""CLI: ask the local knowledge base a question.

Usage:
    python -m docagent.ask "How do I declare a path parameter with a type?"
    python -m docagent.ask --trace "..."     # also print the retrieval trace
"""

import argparse

from dotenv import load_dotenv

from docagent.agent import docagent


def _print_trace(result: dict) -> None:
    trace = result.get("trace") or []
    if not trace:
        return
    print("=== trace ===")
    for i, t in enumerate(trace, 1):
        if t.get("step") == "search_docs":
            print(f"  {i}. search_docs  query={t.get('query')!r}")
        elif t.get("step") == "intent":
            print(f"  {i}. intent       decision={t.get('decision')}")
        else:
            print(f"  {i}. {t.get('step')}  {t}")
    print()


def _print_result(result: dict) -> None:
    """Pretty-print the final Answer (with citations) or the refusal."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if tc["name"] == "Answer":
                    args = tc["args"]
                    print("=== Answer ===")
                    print(args.get("answer", "").strip())
                    citations = args.get("citations", []) or []
                    if citations:
                        print("\n=== Citations ===")
                        for c in citations:
                            print(f"- {c}")
                    return
    if messages:
        print(str(messages[-1].content).strip())


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ask the local document knowledge base a question."
    )
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument(
        "--trace", action="store_true", help="Print the agent's retrieval trace."
    )
    args = parser.parse_args()

    result = docagent.invoke(
        {"question_input": {"question": args.question}},
        config={"recursion_limit": 12},
    )

    if args.trace:
        _print_trace(result)
    _print_result(result)


if __name__ == "__main__":
    main()
