"""Tool registry for the document QA agent."""

from typing import Dict, List

from langchain_core.tools import BaseTool


def get_tools(tool_names: List[str] | None = None) -> List[BaseTool]:
    """Return the agent's tools, or a named subset if ``tool_names`` is given."""
    from docagent.tools.retrieval_tools import (
        Answer,
        Question,
        list_sources,
        search_docs,
    )

    all_tools = {
        "search_docs": search_docs,
        "list_sources": list_sources,
        "Answer": Answer,
        "Question": Question,
    }

    if tool_names is None:
        return list(all_tools.values())
    return [all_tools[name] for name in tool_names if name in all_tools]


def get_tools_by_name(tools: List[BaseTool] | None = None) -> Dict[str, BaseTool]:
    """Return a dict mapping each tool's name to the tool."""
    if tools is None:
        tools = get_tools()
    return {tool.name: tool for tool in tools}
