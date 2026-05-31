#!/usr/bin/env python
"""Ingest local documents (Markdown / text / PDF) into the Chroma knowledge base.

Each chunk is stored with precise provenance metadata (source file + line range,
or page number for PDFs) and a stable ``chunk_id`` used as the Chroma id, so the
hybrid retriever can fuse dense/BM25 results and cite exact locations.

Usage:
    python -m docagent.ingest --path ./corpus/fastapi
    python -m docagent.ingest --path ./corpus/fastapi --reset
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
SKIP_NAMES = {"LICENSE", "SOURCE.md"}  # corpus attribution files, not content


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
        if not f.is_file() or f.name in SKIP_NAMES:
            continue
        ext = f.suffix.lower()
        if ext in TEXT_EXTS:
            docs.extend(_load_text(f))
        elif ext in PDF_EXTS:
            docs.extend(_load_pdf(f))
    return docs


def chunk_documents(
    raw_docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Split docs and attach chunk_id + line range provenance to each chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks: List[Document] = []
    per_source: dict = {}
    for doc in raw_docs:
        text = doc.page_content
        source = doc.metadata.get("source", "unknown")
        for ch in splitter.split_documents([doc]):
            start_index = ch.metadata.get("start_index", 0)
            start_line = text[:start_index].count("\n") + 1
            end_line = start_line + ch.page_content.count("\n")
            idx = per_source.get(source, 0)
            per_source[source] = idx + 1
            ch.metadata["chunk_id"] = f"{source}::{idx}"
            ch.metadata["start_line"] = start_line
            ch.metadata["end_line"] = end_line
            chunks.append(ch)
    return chunks


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

    chunks = chunk_documents(raw_docs, args.chunk_size, args.chunk_overlap)
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

    ids = [c.metadata["chunk_id"] for c in chunks]
    vs.add_documents(chunks, ids=ids)
    print(
        f"Ingested {len(chunks)} chunks into collection "
        f"'{args.collection}' at {args.chroma_path}."
    )
    print("Done.")


if __name__ == "__main__":
    main()
