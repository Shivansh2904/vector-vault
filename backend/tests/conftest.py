"""Shared fixtures for the backend tests.

main.py builds the embedder and the document store when it is imported. The
app is imported once per session (so the model loads once), and every test
then gets an empty store in its own tmp_path. Tests never read or write the
app's real data directory, and no test can see another test's uploads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def app_module(tmp_path_factory):
    # The module-level store in main.py reads DATA_DIR at import time, so point
    # it at a throwaway directory before importing.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("import-data")))
        import main
    return main


@pytest.fixture
def store(app_module, tmp_path, monkeypatch):
    from store import DocumentStore

    fresh = DocumentStore(dimension=app_module.embedder.dimension, data_dir=tmp_path)
    monkeypatch.setattr(app_module, "store", fresh)
    return fresh


@pytest.fixture
def client(app_module, store):
    return TestClient(app_module.app)


@pytest.fixture
def upload(client):
    """Upload a text document and return the response body."""

    def _upload(filename: str, text: str) -> dict:
        files = {"file": (filename, text.encode("utf-8"), "text/plain")}
        r = client.post("/documents/upload", files=files)
        assert r.status_code == 200, r.text
        return r.json()

    return _upload
