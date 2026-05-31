#!/usr/bin/env python
"""CLI: ask the local knowledge base a question.

Usage:
    python -m docagent.ask "What vector store does docagent use?"
"""

import argparse

from dotenv import load_dotenv

from docagent.agent import docagent


def _print_result(result: dict) -> None:
    """Pretty-print the final Answer (with citations) or the refusal."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if tc["name"] == "Answer":
                    args = tc["args"]
                    print("\n=== Answer ===")
                    print(args.get("answer", "").strip())
                    citations = args.get("citations", []) or []
                    if citations:
                        print("\n=== Citations ===")
                        for c in citations:
                            print(f"- {c}")
                    return
    # Out-of-scope (or no Answer produced): show the last message text
    if messages:
        print("\n" + str(messages[-1].content).strip())


def main():
    load_dotenv(".env")
    parser = argparse.ArgumentParser(
        description="Ask the local document knowledge base a question."
    )
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args()

    result = docagent.invoke({"question_input": {"question": args.question}})
    _print_result(result)


if __name__ == "__main__":
    main()
