# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Weekly Dependabot updates for pip (`/backend`), npm (`/frontend`), and GitHub Actions
- `CONTRIBUTING.md` covering backend and frontend setup, plus how to add new file formats

## [1.2.0] — 2026-05-27

### Added
- **Two-stage retrieval** with cross-encoder reranking: when `rerank=true` on `/search`, the bi-encoder retrieves `top_k * candidates_multiplier` candidates and a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores them for higher accuracy
- `candidates_multiplier` parameter (1–10) controls how many initial candidates to fetch
- `rerank_score` field on `SearchResult` when reranking is enabled
- `backend/reranker.py` with lazy-loaded `Reranker` class

## [1.1.0] — 2026-05-27

### Added
- `GET /documents/{doc_id}` — return metadata for a single document
- `GET /documents/{doc_id}/chunks` — inspect the text chunks a document was split into
- DOCX support via `python-docx` (alongside existing PDF, TXT, MD)
- Backend test suite (`backend/tests/test_api.py`) with 9 tests covering health, list, upload, search, get, get-chunks, delete

### Fixed
- CI: added `frontend/src/vite-env.d.ts` for `import.meta.env` typing
- CI: added `frontend/package-lock.json` for npm cache

## [1.0.0] — 2026-05-17

### Added
- Initial release
- FastAPI backend with FAISS FlatL2 index + sentence-transformers (`all-MiniLM-L6-v2`)
- Document upload (PDF, TXT, MD), chunking, embedding, semantic search
- React + TypeScript frontend (upload, library, search UI)
- Thread-safe `DocumentStore` with `threading.Lock` on all mutations
- 20 MB file size limit, fetch timeouts (10s default, 30s uploads)
- Docker Compose, GitHub Actions CI
