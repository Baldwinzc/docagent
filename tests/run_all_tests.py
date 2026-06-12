#!/usr/bin/env python
"""Run the citelocal_agent test suite.

Examples:
    python tests/run_all_tests.py            # local retrieval tests only (no key)
    python tests/run_all_tests.py --all      # also run the LLM end-to-end tests
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run citelocal_agent tests")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include the LLM end-to-end tests (requires an API key).",
    )
    args = parser.parse_args()

    os.environ.setdefault("LANGSMITH_PROJECT", "citelocal_agent-eval")

    test_files = ["tests/test_retrieval.py"]
    if args.all:
        test_files.append("tests/test_response.py")

    cmd = [
        "python",
        "-m",
        "pytest",
        *test_files,
        "-v",
        "--disable-warnings",
        "--agent-module=agent",
    ]

    project_root = Path(__file__).parent.parent
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
