"""Bulk-index every supported file in a directory, then run a semantic search.

Start the backend first:
    cd backend && uvicorn main:app --reload

Then:
    python examples/bulk_index_and_search.py path/to/docs/ "your search query"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

API = os.environ.get("VECTORVAULT_API", "http://localhost:8000")
SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


def index_directory(client: httpx.Client, directory: Path) -> list[dict]:
    """Upload every supported file in directory to /documents/upload."""
    indexed = []
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files found in {directory}. Looking for: {sorted(SUPPORTED)}")
        return []

    print(f"Indexing {len(files)} files from {directory}...")
    for path in files:
        with path.open("rb") as fh:
            try:
                r = client.post(
                    f"{API}/documents/upload",
                    files={"file": (path.name, fh, "application/octet-stream")},
                    timeout=60,
                )
                r.raise_for_status()
                data = r.json()
                indexed.append(data)
                print(f"  [{data['chunk_count']:>3} chunks] {path.name}")
            except httpx.HTTPStatusError as exc:
                print(f"  [SKIP] {path.name} - {exc.response.status_code}: {exc.response.text[:100]}")

    return indexed


def search(client: httpx.Client, query: str, top_k: int = 5, rerank: bool = False) -> None:
    print(f"\nSearching for: {query!r} (top_k={top_k}, rerank={rerank})")
    body = {"query": query, "top_k": top_k, "rerank": rerank}
    if rerank:
        body["candidates_multiplier"] = 3

    r = client.post(f"{API}/search", json=body, timeout=30)
    r.raise_for_status()
    results = r.json()["results"]

    if not results:
        print("  (no results)")
        return

    for i, hit in enumerate(results, 1):
        score_info = f"score={hit['score']:.4f}"
        if hit.get("rerank_score") is not None:
            score_info += f" rerank={hit['rerank_score']:+.4f}"
        snippet = hit["chunk_text"][:200].replace("\n", " ")
        print(f"  {i}. {hit['doc_filename']:<30} {score_info}")
        print(f"     {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory of documents to index")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranking")
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    with httpx.Client() as client:
        # Check health first
        try:
            client.get(f"{API}/health", timeout=5).raise_for_status()
        except httpx.RequestError:
            print(f"Cannot reach {API}. Start the backend: cd backend && uvicorn main:app --reload", file=sys.stderr)
            sys.exit(1)

        index_directory(client, directory)
        search(client, args.query, top_k=args.top_k, rerank=args.rerank)


if __name__ == "__main__":
    main()
