# VectorVault

[![CI](https://img.shields.io/github/actions/workflow/status/Shivansh2904/vector-vault/ci.yml?style=flat-square&label=CI)](https://github.com/Shivansh2904/vector-vault/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

Semantic search over your own documents. Upload PDF, text, Markdown or Word
files, then search them in plain English. VectorVault splits each document into
chunks, embeds them with a small sentence-transformers model and finds the
chunks nearest to your query in a FAISS index. An optional cross-encoder can
re-rank the best candidates.

It runs on your own machine and needs no API keys. The two models are
downloaded from Hugging Face the first time they are used, and the UI loads
its fonts from Google Fonts.

The backend is FastAPI; the UI is React, TypeScript and Tailwind.

## How it works

```
upload ─▶ extract text ─▶ chunk (≤ 200 tokens) ─▶ all-MiniLM-L6-v2 ─▶ FAISS IndexFlatL2
                                                                       (index.faiss + metadata.json)

query ─▶ all-MiniLM-L6-v2 ─▶ exact nearest chunks ─▶ optional cross-encoder rerank ─▶ results
```

**Indexing**

1. **Text extraction.** PDFs go through `pypdf` and Word files through
   `python-docx` (paragraph text only, so tables are skipped). `.txt` and `.md`
   are decoded as UTF-8, falling back to Latin-1.
2. **Chunking** (`backend/chunker.py`). Text is split into chunks of at most
   200 tokens, with about 40 tokens of overlap, counted by the embedding
   model's own tokenizer. Chunks break between words. The limit matters because
   all-MiniLM-L6-v2 reads 256 tokens per input and silently drops the rest, so
   anything past that point in a chunk would never be searchable.
   `backend/tests/test_chunking.py` checks every chunk against the model's
   limit and hides a sentence deep in a document to make sure search finds it.
3. **Embedding.** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions),
   normalised to unit length.
4. **Index** (`backend/store.py`). A FAISS `IndexFlatL2`, which is exact
   search. For unit vectors, ranking by L2 distance is the same as ranking by
   cosine similarity. The index and a JSON metadata file are written to
   `DATA_DIR` after every upload and delete.

**Search**

1. The query is embedded with the same model and FAISS returns the nearest
   chunks.
2. Each result's `score` is `1 − d²/4`, which for unit vectors equals
   `(1 + cosine) / 2`. An unrelated chunk still scores around 0.5, so compare
   scores with each other rather than reading them as percentages.
3. With `"rerank": true`, the API fetches `top_k × candidates_multiplier`
   candidates (at most 100) and scores each (query, chunk) pair with
   `cross-encoder/ms-marco-MiniLM-L-6-v2`, which reads the query and the chunk
   together. Results come back ordered by `rerank_score`. This is slower than
   plain search, because every candidate goes through the model.

## Quick start

### Local

Needs Python 3.11 and Node.js 18 or later.

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The first start downloads the embedding model from Hugging Face.

Frontend, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The API's interactive docs are at
http://localhost:8000/docs.

### Docker Compose

```bash
docker compose up --build
```

The UI is at http://localhost:5173 and the API at http://localhost:8000. Both
ports are bound to 127.0.0.1. The index is kept in `./data`. The backend image
installs CPU-only PyTorch and downloads the embedding model during the build,
so the first build takes a while.

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload a file (multipart field `file`: `.pdf`, `.txt`, `.md` or `.docx`). Returns `doc_id`, `filename`, `chunk_count`. |
| `GET` | `/documents` | List indexed documents. |
| `GET` | `/documents/{doc_id}` | Metadata for one document, or 404. |
| `GET` | `/documents/{doc_id}/chunks` | The chunks a document was split into. |
| `DELETE` | `/documents/{doc_id}` | Delete a document. Its vectors stay in the FAISS index but no longer appear in results (see Limitations). |
| `POST` | `/search` | Body: `{"query": "...", "top_k": 5, "rerank": false, "candidates_multiplier": 3}`. `top_k` is 1–20 and `candidates_multiplier` 1–10. |
| `GET` | `/health` | Status plus document, chunk and index-vector counts. |

```bash
curl -X POST http://localhost:8000/documents/upload -F "file=@report.pdf"

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what are the key risks?", "top_k": 3}'

# the same search, re-ranked by the cross-encoder
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what are the key risks?", "top_k": 3, "rerank": true}'
```

`examples/bulk_index_and_search.py` uploads every supported file in a folder
and then runs one query against them.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/
```

The tests use the real embedding model and cross-encoder (downloaded on the
first run), and each test gets an empty index in its own temporary directory. CI runs them, plus a type check
and production build of the frontend.

With the virtual environment active, `make test` does the same from the
repository root; `make help` lists the other targets.

## Configuration

| Variable | When it is read | Default | Meaning |
|---|---|---|---|
| `DATA_DIR` | backend start-up | `data` | Where `index.faiss` and `metadata.json` are stored |
| `CORS_ORIGINS` | backend start-up | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated origins whose pages may read API responses in a browser. Set it if the UI is served from another address. An empty value allows none. |
| `VITE_API_URL` | when the frontend is built or the dev server starts | `http://localhost:8000` | API address the UI calls |

## Limitations

- **No authentication and no upload size limit.** Anyone who can reach the
  API can upload, read and delete documents, and each upload is read into
  memory whole. Keep it on localhost. Docker Compose binds it to 127.0.0.1.
- **CORS limits reading, not sending.** Browsers only let pages from
  `CORS_ORIGINS` read API responses. A page on another site can still send
  a request that needs no CORS preflight, such as a form upload to
  `/documents/upload`; it just cannot read the reply.
- **Uploads block the server.** Parsing and embedding run inside `async`
  routes, so other requests wait until a large upload has finished.
- **Delete is a soft delete.** Deleted vectors stay in `index.faiss` and are
  filtered out of results. Uploading the same file twice indexes it twice.
- **Metadata is one JSON file.** `metadata.json` holds the text of every
  chunk and is rewritten on each upload and delete, so writes slow down as the
  corpus grows. Writes go to temporary files that are renamed into place; if
  the files in `DATA_DIR` cannot be read at start-up, the API refuses to start
  rather than overwrite them.
- **Formats.** Word tables are skipped, and scanned PDFs without a text layer
  are rejected. The UI's file picker is filtered to PDF, TXT and Markdown, so
  upload `.docx` files through the API. The UI has no switch for re-ranking.
- **No measured retrieval quality or latency yet.** There is no evaluation set
  or benchmark script in the repository, so this README does not quote
  accuracy or speed figures.

## Project layout

```
backend/
  main.py        FastAPI app, routes and schemas
  chunker.py     token-based chunking
  embedder.py    sentence-transformers wrapper
  reranker.py    cross-encoder re-ranking (loaded on first use)
  store.py       FAISS index and metadata, saved to DATA_DIR
  tests/         pytest suite
frontend/        React + TypeScript + Vite UI, served by nginx in Docker
examples/        bulk indexing script
```

## License

MIT © 2026 Shivansh Mishra
