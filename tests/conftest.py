#!/usr/bin/env python

import sys
from pathlib import Path

import pytest

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--agent-module",
        action="store",
        default="agent",
        help="Which docagent graph module to test (default: agent)",
    )


@pytest.fixture(scope="session")
def agent_module_name(request):
    """Return the agent module name from the command line."""
    return request.config.getoption("--agent-module")
