"""Tool registry helpers for the document QA agent."""

from typing import Dict, List

from langchain_core.tools import BaseTool


def get_tools_by_name(tools: List[BaseTool]) -> Dict[str, BaseTool]:
    """Return a dict mapping each tool's name to the tool."""
    return {tool.name: tool for tool in tools}
