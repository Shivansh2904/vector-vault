"""Persistence tests for DocumentStore, with small hand-made vectors (no model)."""

from __future__ import annotations

import numpy as np
import pytest

import store as store_module
from store import DocumentStore

DIM = 4


def vec(i: int) -> np.ndarray:
    v = np.zeros((1, DIM), dtype=np.float32)
    v[0, i] = 1.0
    return v


def snapshot(directory) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def test_documents_survive_a_restart(tmp_path):
    s = DocumentStore(dimension=DIM, data_dir=tmp_path)
    s.add_document("a.txt", ["alpha"], vec(0))

    reopened = DocumentStore(dimension=DIM, data_dir=tmp_path)
    [hit] = reopened.search(vec(0), top_k=1)
    assert hit["doc_filename"] == "a.txt"
    assert hit["chunk_text"] == "alpha"


def test_unreadable_metadata_is_an_error_not_an_empty_index(tmp_path):
    s = DocumentStore(dimension=DIM, data_dir=tmp_path)
    s.add_document("a.txt", ["alpha"], vec(0))
    meta = tmp_path / "metadata.json"
    meta.write_bytes(meta.read_bytes() + b"x")
    before = snapshot(tmp_path)

    with pytest.raises(RuntimeError, match="metadata.json"):
        DocumentStore(dimension=DIM, data_dir=tmp_path)
    assert snapshot(tmp_path) == before


def test_a_failed_save_leaves_the_previous_files_intact(tmp_path, monkeypatch):
    s = DocumentStore(dimension=DIM, data_dir=tmp_path)
    s.add_document("a.txt", ["alpha"], vec(0))
    before = snapshot(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    # Fail while writing the metadata, after the index has been written.
    monkeypatch.setattr(store_module.json, "dumps", fail)
    with pytest.raises(OSError):
        s.add_document("b.txt", ["bravo"], vec(1))
    monkeypatch.undo()

    assert {k: v for k, v in snapshot(tmp_path).items() if not k.endswith(".tmp")} == before


def test_a_crash_between_the_two_writes_does_not_mislabel_results(tmp_path):
    s = DocumentStore(dimension=DIM, data_dir=tmp_path)
    s.add_document("a.txt", ["alpha"], vec(0))
    committed_metadata = (tmp_path / "metadata.json").read_bytes()
    s.add_document("b.txt", ["bravo"], vec(1))
    # As if the process died after index.faiss was written but before
    # metadata.json was: the index holds b's vector, the metadata does not.
    (tmp_path / "metadata.json").write_bytes(committed_metadata)

    s = DocumentStore(dimension=DIM, data_dir=tmp_path)
    assert [d["filename"] for d in s.list_documents()] == ["a.txt"]
    s.add_document("c.txt", ["charlie"], vec(2))

    [hit] = s.search(vec(2), top_k=1)
    assert (hit["doc_filename"], hit["score"]) == ("c.txt", 1.0)
    # b was never committed, so nothing stored should match its vector exactly.
    [hit] = s.search(vec(1), top_k=1)
    assert hit["score"] < 1.0, f"b's vector came back as {hit['doc_filename']}"
    assert s.get_stats()["index_vectors"] == 2
