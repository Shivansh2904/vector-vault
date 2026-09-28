def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_documents_empty(client):
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json() == []


def test_search_empty_index(client):
    r = client.post("/search", json={"query": "test"})
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_upload_txt(upload):
    data = upload("test.txt", "This is a test document about machine learning and neural networks.")
    assert data["filename"] == "test.txt"
    assert data["chunk_count"] >= 1


def test_get_document(client, upload):
    doc_id = upload("gettest.txt", "Document for get test. " * 20)["doc_id"]

    r = client.get(f"/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["doc_id"] == doc_id


def test_get_nonexistent_document(client):
    r = client.get("/documents/nonexistent-id")
    assert r.status_code == 404


def test_delete_document(client, upload):
    doc_id = upload("delete.txt", "Document to delete. " * 20)["doc_id"]

    r = client.delete(f"/documents/{doc_id}")
    assert r.status_code == 200

    r = client.get(f"/documents/{doc_id}")
    assert r.status_code == 404


def test_get_document_chunks(client, upload):
    doc_id = upload("chunktest.txt", "Chunk test content. " * 50)["doc_id"]

    r = client.get(f"/documents/{doc_id}/chunks")
    assert r.status_code == 200
    data = r.json()
    assert data["doc_id"] == doc_id
    assert data["filename"] == "chunktest.txt"
    assert len(data["chunks"]) >= 1
    assert all("text" in c and "chunk_index" in c for c in data["chunks"])


def test_get_chunks_nonexistent(client):
    r = client.get("/documents/nonexistent/chunks")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Two-stage retrieval (these load the real cross-encoder)
# ---------------------------------------------------------------------------

# The bi-encoder puts the Karloff sentence first because it shares the word
# "Frankenstein" with the query. The cross-encoder reads query and passage
# together and puts the sentence that answers the question first.
FRANKENSTEIN_QUERY = "who wrote the novel Frankenstein"
FRANKENSTEIN_DOCS = {
    "karloff.txt": "Frankenstein's monster has been played on film by many actors, including Boris Karloff.",
    "dracula.txt": "The novel Dracula was published by Bram Stoker in 1897.",
    "shelley.txt": "Mary Shelley began writing the story in 1816 during a wet summer at Lake Geneva.",
}


def test_search_without_rerank_returns_bi_encoder_order(client, upload):
    for name, text in FRANKENSTEIN_DOCS.items():
        upload(name, text)

    r = client.post("/search", json={"query": FRANKENSTEIN_QUERY, "top_k": 3})
    results = r.json()["results"]
    assert [x["doc_filename"] for x in results] == ["karloff.txt", "dracula.txt", "shelley.txt"]
    assert all(x["rerank_score"] is None for x in results)


def test_search_with_rerank_rescores_the_candidate_pool(client, upload):
    for name, text in FRANKENSTEIN_DOCS.items():
        upload(name, text)

    # top_k=1 with a multiplier of 3: the answer is third for the bi-encoder,
    # so it can only come back if all three candidates were reranked.
    r = client.post("/search", json={
        "query": FRANKENSTEIN_QUERY,
        "top_k": 1,
        "rerank": True,
        "candidates_multiplier": 3,
    })
    assert r.status_code == 200
    [hit] = r.json()["results"]
    assert hit["doc_filename"] == "shelley.txt"
    assert isinstance(hit["rerank_score"], float)


def test_rerank_results_are_ordered_by_rerank_score(client, upload):
    for name, text in FRANKENSTEIN_DOCS.items():
        upload(name, text)

    r = client.post("/search", json={"query": FRANKENSTEIN_QUERY, "top_k": 3, "rerank": True})
    results = r.json()["results"]
    scores = [x["rerank_score"] for x in results]
    assert len(results) == 3
    assert scores == sorted(scores, reverse=True)
    assert results[0]["doc_filename"] == "shelley.txt"
    assert results[-1]["doc_filename"] == "karloff.txt"


# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


def test_uploads_are_written_to_the_tests_own_directory(upload, tmp_path):
    upload("isolated.txt", "Written to this test's directory only.")
    assert (tmp_path / "index.faiss").is_file()
    assert "isolated.txt" in (tmp_path / "metadata.json").read_text(encoding="utf-8")


def test_each_test_starts_with_an_empty_index(client):
    # Runs after tests that uploaded documents; none of them should be visible.
    assert client.get("/documents").json() == []
    assert client.get("/health").json()["index_vectors"] == 0
