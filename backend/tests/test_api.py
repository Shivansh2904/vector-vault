from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_list_documents_empty():
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json() == []

def test_search_empty_index():
    r = client.post("/search", json={"query": "test"})
    assert r.status_code == 200
    assert r.json()["results"] == []

def test_upload_txt():
    content = b"This is a test document about machine learning and neural networks."
    files = {"file": ("test.txt", content, "text/plain")}
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "test.txt"
    assert data["chunk_count"] >= 1
    return data["doc_id"]

def test_get_document():
    # upload first
    content = b"Document for get test. " * 20
    files = {"file": ("gettest.txt", content, "text/plain")}
    r = client.post("/documents/upload", files=files)
    doc_id = r.json()["doc_id"]

    r2 = client.get(f"/documents/{doc_id}")
    assert r2.status_code == 200
    assert r2.json()["doc_id"] == doc_id

def test_get_nonexistent_document():
    r = client.get("/documents/nonexistent-id")
    assert r.status_code == 404

def test_delete_document():
    content = b"Document to delete. " * 20
    files = {"file": ("delete.txt", content, "text/plain")}
    r = client.post("/documents/upload", files=files)
    doc_id = r.json()["doc_id"]

    r2 = client.delete(f"/documents/{doc_id}")
    assert r2.status_code == 200

    r3 = client.get(f"/documents/{doc_id}")
    assert r3.status_code == 404
