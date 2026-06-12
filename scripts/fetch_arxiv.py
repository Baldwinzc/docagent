#!/usr/bin/env python
"""Download arXiv papers (PDF) into ./papers for fully-local Q&A.

Papers are saved locally only — ``papers/`` is gitignored and nothing is ever
uploaded to a third party (that's the whole point versus cloud paper tools).

    python scripts/fetch_arxiv.py 1706.03762 2005.11401
    python scripts/fetch_arxiv.py --demo        # a small starter set
    # then:
    python -m citelocal_agent.ingest --path ./papers --reset
"""

import argparse
import urllib.request
from pathlib import Path

PAPERS = Path(__file__).resolve().parent.parent / "papers"

# A small, recognizable starter set (downloaded locally, not redistributed).
# These cover the full_corpus eval cases in src/citelocal_agent/eval/data/qa_cases.jsonl.
DEMO = {
    "1706.03762": "attention-is-all-you-need",
    "2005.11401": "retrieval-augmented-generation",
    "1810.04805": "bert",
    "1910.10683": "t5",
    "1907.11692": "roberta",
    "2004.04906": "dense-passage-retrieval",
    "1908.10084": "sentence-bert",
    "2005.14165": "gpt-3",
}


def fetch(arxiv_id: str, name: str | None = None) -> None:
    PAPERS.mkdir(exist_ok=True)
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    fname = f"{name or arxiv_id}.pdf"
    req = urllib.request.Request(url, headers={"User-Agent": "citelocal_agent-arxiv"})
    data = urllib.request.urlopen(req, timeout=60).read()
    (PAPERS / fname).write_bytes(data)
    print(f"  {arxiv_id} -> papers/{fname} ({len(data) // 1024} KB)")


def main():
    p = argparse.ArgumentParser(description="Download arXiv PDFs into ./papers")
    p.add_argument("ids", nargs="*", help="arXiv ids, e.g. 1706.03762")
    p.add_argument("--demo", action="store_true", help="download the starter set")
    args = p.parse_args()

    items = list(DEMO.items()) if args.demo else [(i, None) for i in args.ids]
    if not items:
        p.error("give one or more arXiv ids, or use --demo")

    print(f"Downloading {len(items)} paper(s) into {PAPERS} ...")
    for aid, name in items:
        try:
            fetch(aid, name)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {aid}: {e}")
    print("Done.\nNext: python -m citelocal_agent.ingest --path ./papers --reset")


if __name__ == "__main__":
    main()
