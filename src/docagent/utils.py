"""Utility helpers for the document QA agent.

Trimmed down from the original email-assistant utilities to keep only the
message helpers that the agent and tests actually use.
"""

from typing import List, Any


def extract_message_content(message) -> str:
    """Extract content from different message types as a clean string."""
    content = message.content

    if isinstance(content, str) and "<Recursion on AIMessage with id=" in content:
        return "[Recursive content]"

    if isinstance(content, str):
        return content

    # AIMessage list-of-parts format
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "\n".join(text_parts)

    return str(content)


def extract_tool_calls(messages: List[Any]) -> List[str]:
    """Extract tool-call names (lower-cased) from a list of messages."""
    tool_call_names = []
    for message in messages:
        if isinstance(message, dict) and message.get("tool_calls"):
            tool_call_names.extend(call["name"].lower() for call in message["tool_calls"])
        elif hasattr(message, "tool_calls") and message.tool_calls:
            tool_call_names.extend(call["name"].lower() for call in message.tool_calls)
    return tool_call_names


def format_messages_string(messages: List[Any]) -> str:
    """Format messages into a single string for analysis / evaluation."""
    return "\n".join(message.pretty_repr() for message in messages)


def show_graph(graph, xray=False):
    """Render a LangGraph mermaid diagram (for use inside notebooks/IPython)."""
    from IPython.display import Image

    return Image(graph.get_graph(xray=xray).draw_mermaid_png())
