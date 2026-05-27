# Contributing to VectorVault

Thanks for considering a contribution! VectorVault is split between a Python backend (FAISS + sentence-transformers) and a React frontend.

## Working on the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# Swagger: http://localhost:8000/docs
```

Tests:

```bash
cd backend
pip install pytest httpx
pytest tests/ -v
```

### Backend architecture

- `main.py` — FastAPI app, all routes, Pydantic schemas
- `embedder.py` — sentence-transformers wrapper (bi-encoder)
- `reranker.py` — cross-encoder for two-stage retrieval
- `store.py` — FAISS index + in-memory metadata
- `chunker.py` — token-aware text chunking

When adding a new endpoint, place static-path routes (like `/export`, `/stats`) BEFORE dynamic-path routes (`/{doc_id}`) — FastAPI matches in registration order.

## Working on the frontend

```bash
cd frontend
npm install
npm run dev
# UI: http://localhost:5173
```

Set `VITE_API_URL` in `.env` to point at your backend.

## Adding a new file format to upload

1. Add the extension to `_SUPPORTED_EXTENSIONS` in `main.py`
2. Add a branch in `_extract_text()` that returns plain text
3. Add a test in `backend/tests/test_api.py` that uploads a sample file of that type
4. Update README

## Style

- Python: PEP 8, type hints, Pydantic v2 schemas
- TypeScript: strict mode
- All search/store operations should be threadsafe (use the existing patterns in `store.py`)

## Submitting a PR

1. Fork, branch, commit
2. Tests pass in both backend and frontend
3. Update README if you add an endpoint or upload format

## License

By contributing, you agree your contributions are licensed under MIT.
