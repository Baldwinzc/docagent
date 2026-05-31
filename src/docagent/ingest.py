#!/usr/bin/env python
"""Ingest local documents (Markdown / text / PDF) into the Chroma knowledge base.

Usage:
    python -m docagent.ingest --path ./sample_docs
    python -m docagent.ingest --path ./sample_docs --reset
"""

import argparse
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from docagent.configuration import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from docagent.vectorstore import get_vectorstore

load_dotenv()

TEXT_EXTS = {".md", ".markdown", ".txt", ".rst"}
PDF_EXTS = {".pdf"}


def _load_text(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_pdf(path: Path) -> List[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": i + 1},
                )
            )
    return docs


def load_documents(root: Path) -> List[Document]:
    """Load all supported files under ``root`` (recursively if a directory)."""
    docs: List[Document] = []
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in TEXT_EXTS:
            docs.extend(_load_text(f))
        elif ext in PDF_EXTS:
            docs.extend(_load_pdf(f))
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Chroma knowledge base."
    )
    parser.add_argument("--path", required=True, help="File or directory to ingest.")
    parser.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--reset", action="store_true", help="Clear the collection before ingesting."
    )
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    print(f"Loading documents from {root} ...")
    raw_docs = load_documents(root)
    if not raw_docs:
        raise SystemExit("No supported documents found (.md/.markdown/.txt/.rst/.pdf).")
    print(f"Loaded {len(raw_docs)} raw document section(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks.")

    vs = get_vectorstore(
        persist_directory=args.chroma_path, collection_name=args.collection
    )

    if args.reset:
        try:
            existing = vs.get()
            ids = existing.get("ids", [])
            if ids:
                vs.delete(ids=ids)
                print(f"Reset: removed {len(ids)} existing chunks.")
        except Exception as e:  # noqa: BLE001
            print(f"Reset skipped: {e}")

    vs.add_documents(chunks)
    print(
        f"Ingested {len(chunks)} chunks into collection "
        f"'{args.collection}' at {args.chroma_path}."
    )
    print("Done.")


if __name__ == "__main__":
    main()
