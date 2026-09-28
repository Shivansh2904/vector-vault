"""
store.py — FAISS-backed document store for VectorVault.

Manages the mapping between logical documents (with metadata) and the
underlying FAISS flat-L2 index.  Persistence is handled by writing the
FAISS index to ``<data_dir>/index.faiss`` and the metadata dict to
``<data_dir>/metadata.json`` on every mutating operation.  ``data_dir``
defaults to the ``DATA_DIR`` environment variable, or ``data``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

log = logging.getLogger("vectorvault.store")


class StoreLoadError(RuntimeError):
    """The files in the data directory exist but cannot be used."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DocumentStore
# ---------------------------------------------------------------------------

class DocumentStore:
    """In-memory document store backed by a FAISS FlatL2 index.

    Each document is split into chunks before indexing.  Every chunk
    occupies exactly one row in the FAISS index.  A separate metadata
    dict maps each FAISS row index (``chunk_id``) back to the owning
    document.

    Thread-safety note: the current implementation is intentionally
    single-threaded.  FastAPI's default async event loop executes
    synchronous code in a thread pool, so for a local prototype this
    is fine.  Production use would add an ``asyncio.Lock``.
    """

    def __init__(
        self,
        dimension: int = 384,
        data_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._dimension = dimension
        self._data_dir = Path(data_dir if data_dir is not None else os.getenv("DATA_DIR", "data"))
        self._index_path = self._data_dir / "index.faiss"
        self._meta_path = self._data_dir / "metadata.json"

        # doc_id → {filename, chunk_count, created_at, chunk_ids: [int, …]}
        self._docs: dict[str, dict[str, Any]] = {}

        # FAISS row index → doc_id
        self._chunk_to_doc: dict[int, str] = {}

        # FAISS row index → chunk text
        self._chunk_texts: dict[int, str] = {}

        # Next available FAISS row counter (monotonically increasing; we
        # never actually remove rows from FAISS — deletion is handled by
        # filtering results).
        self._next_row: int = 0

        # Rows that have been "logically" deleted (their document was removed).
        self._deleted_rows: set[int] = set()

        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)

        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist FAISS index and metadata to disk.

        Both files are written under a temporary name and then renamed over
        the old ones, index first, so a failed write leaves the previous
        files in place. Renaming metadata.json is the commit point: if the
        process dies between the two renames, the index has rows at the end
        that metadata.json does not know about, and ``_load`` drops them.
        """
        payload = {
            "next_row": self._next_row,
            "deleted_rows": list(self._deleted_rows),
            "docs": self._docs,
            "chunk_to_doc": {str(k): v for k, v in self._chunk_to_doc.items()},
            "chunk_texts": {str(k): v for k, v in self._chunk_texts.items()},
        }
        index_tmp = self._index_path.with_name(self._index_path.name + ".tmp")
        meta_tmp = self._meta_path.with_name(self._meta_path.name + ".tmp")
        faiss.write_index(self._index, str(index_tmp))
        meta_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(index_tmp, self._index_path)
        os.replace(meta_tmp, self._meta_path)

    def _load(self) -> None:
        """Reload persisted index and metadata from disk (if present).

        Raises StoreLoadError rather than starting empty when the files are
        there but unusable, so a later save cannot overwrite them.
        """
        if not self._meta_path.exists():
            if self._index_path.exists():
                log.warning(
                    "%s exists without %s; starting with an empty index",
                    self._index_path, self._meta_path.name,
                )
            return

        try:
            payload = json.loads(self._meta_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(self._index_path))
        except Exception as exc:
            raise StoreLoadError(
                f"Could not load {self._meta_path} and {self._index_path}: {exc}. "
                "The files have not been changed; move them aside to start with "
                "an empty index."
            ) from exc

        next_row = payload.get("next_row", 0)
        if index.ntotal < next_row:
            raise StoreLoadError(
                f"{self._index_path} holds {index.ntotal} vectors but "
                f"{self._meta_path} expects {next_row}. The files have not been changed."
            )
        if index.ntotal > next_row:
            # The last save stopped after the index was renamed but before
            # metadata.json was. Rows are only ever appended, so the extra
            # rows at the end are the ones that were never committed.
            log.warning(
                "%s has %d uncommitted vectors from an interrupted save; dropping them",
                self._index_path, index.ntotal - next_row,
            )
            committed = faiss.IndexFlatL2(self._dimension)
            if next_row:
                committed.add(index.reconstruct_n(0, next_row))
            index = committed

        self._index = index
        self._next_row = next_row
        self._deleted_rows = set(payload.get("deleted_rows", []))
        self._docs = payload.get("docs", {})
        self._chunk_to_doc = {int(k): v for k, v in payload.get("chunk_to_doc", {}).items()}
        self._chunk_texts = {int(k): v for k, v in payload.get("chunk_texts", {}).items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(
        self,
        filename: str,
        chunks: list[str],
        embeddings: np.ndarray,
    ) -> str:
        """Index a document and return its new doc_id.

        Parameters
        ----------
        filename:
            Original file name (used for display only).
        chunks:
            Text chunks produced by the chunker.
        embeddings:
            Array of shape ``(len(chunks), dimension)`` of L2-normalised
            float32 vectors.

        Returns
        -------
        str
            UUID4 document identifier.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")

        doc_id = str(uuid.uuid4())
        chunk_ids: list[int] = []

        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            row = self._next_row
            self._index.add(vec.reshape(1, -1))
            self._chunk_to_doc[row] = doc_id
            self._chunk_texts[row] = chunk
            chunk_ids.append(row)
            self._next_row += 1

        self._docs[doc_id] = {
            "filename": filename,
            "chunk_count": len(chunks),
            "chunk_ids": chunk_ids,
            "created_at": _now_iso(),
        }

        self._save()
        return doc_id

    def remove_document(self, doc_id: str) -> None:
        """Logically remove a document from the store.

        FAISS FlatL2 does not support in-place row removal, so we track
        deleted rows and filter them out at search time.
        """
        if doc_id not in self._docs:
            raise KeyError(f"Document {doc_id!r} not found")

        doc = self._docs.pop(doc_id)
        for row in doc["chunk_ids"]:
            self._deleted_rows.add(row)
            self._chunk_to_doc.pop(row, None)
            self._chunk_texts.pop(row, None)

        self._save()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-K most similar chunks for a query embedding.

        Deleted chunks are filtered out; additional candidates are fetched
        if needed to fill the requested top_k.

        Parameters
        ----------
        query_embedding:
            Shape ``(1, dimension)`` float32 array.
        top_k:
            Number of results to return.

        Returns
        -------
        List of dicts with keys: doc_id, doc_filename, chunk_text, score.
        """
        if self._index.ntotal == 0:
            return []

        # Over-fetch to account for deleted rows.
        fetch_k = min(top_k * 4 + len(self._deleted_rows), self._index.ntotal)
        distances, indices = self._index.search(query_embedding.reshape(1, -1), fetch_k)

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx in self._deleted_rows:
                continue
            doc_id = self._chunk_to_doc.get(int(idx))
            if doc_id is None or doc_id not in self._docs:
                continue
            doc = self._docs[doc_id]
            # Convert L2 distance to a similarity score in [0, 1].
            # For L2-normalised vectors: dist ∈ [0, 4]; smaller = more similar.
            score = float(max(0.0, 1.0 - dist / 4.0))
            results.append(
                {
                    "doc_id": doc_id,
                    "doc_filename": doc["filename"],
                    "chunk_text": self._chunk_texts.get(int(idx), ""),
                    "score": round(score, 4),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def list_documents(self) -> list[dict[str, Any]]:
        """Return metadata for all indexed documents."""
        return [
            {
                "doc_id": doc_id,
                "filename": meta["filename"],
                "chunk_count": meta["chunk_count"],
                "created_at": meta["created_at"],
            }
            for doc_id, meta in self._docs.items()
        ]

    def get_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        """Return all chunks for a single document with their text and position.

        Chunks are returned in their original order (the order in which they
        were indexed).  Each entry has a ``chunk_index`` (0-based position
        within the document) and the chunk ``text``.

        Raises
        ------
        KeyError
            If ``doc_id`` is not present in the store.
        """
        if doc_id not in self._docs:
            raise KeyError(f"Document {doc_id!r} not found")

        doc = self._docs[doc_id]
        return [
            {"chunk_index": i, "text": self._chunk_texts.get(row, "")}
            for i, row in enumerate(doc["chunk_ids"])
        ]

    def get_stats(self) -> dict[str, int]:
        """Return aggregate index statistics."""
        return {
            "total_documents": len(self._docs),
            "total_chunks": sum(d["chunk_count"] for d in self._docs.values()),
            "index_vectors": self._index.ntotal,
        }
