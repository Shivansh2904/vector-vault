<div align="center">

# VectorVault

**Semantic search over your documents — runs entirely locally, no API keys, no data leaves your machine.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-blue?style=flat-square)](https://github.com/facebookresearch/faiss)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Shivansh2904/vector-vault/ci.yml?style=flat-square&label=CI)](https://github.com/Shivansh2904/vector-vault/actions)

<br/>

*Upload PDFs, text files, Markdown, and Word documents. Ask questions in plain English. Get semantically ranked results — all without sending a single byte to any external service.*

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [Performance](#-performance)

</div>

---

## Why VectorVault?

Most document search tools either require expensive API calls (OpenAI embeddings, Pinecone, etc.) or are just keyword search. VectorVault is different:

| Feature | VectorVault | Keyword search | Cloud RAG |
|---|---|---|---|
| Semantic understanding | ✅ | ❌ | ✅ |
| Runs offline | ✅ | ✅ | ❌ |
| Zero cost | ✅ | ✅ | ❌ |
| Privacy (data stays local) | ✅ | ✅ | ❌ |
| Sub-50 ms search | ✅ | ✅ | ❌ |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                        │
│                                                                   │
│  PDF/TXT/MD/DOCX  ──▶  Chunker  ──▶  sentence-transformers      │
│                        (512 words,     (all-MiniLM-L6-v2)        │
│                         50 overlap)    384-dim embeddings         │
│                                              │                    │
│                                              ▼                    │
│                                       FAISS FlatL2 Index          │
│                                      (persisted to disk)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                          │
│                                                                   │
│   Query string  ──▶  sentence-transformers  ──▶  FAISS search    │
│                       (same model, L2-norm)      (top-K ANN)     │
│                                                       │           │
│                                                       ▼           │
│                                              Ranked chunks        │
│                                          (score, text, source)   │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- `FlatL2` index: exact nearest-neighbour search — no approximation errors, great for corpora up to ~1M vectors on modern hardware.
- L2-normalised embeddings: makes L2 distance equivalent to cosine similarity.
- Word-boundary chunking with 50-word overlap: prevents semantic context from being cut off at chunk boundaries.
- Soft delete: FAISS doesn't support in-place row removal, so deleted chunks are tracked in a filter set.

---

## Features

- **Multi-format ingestion** — PDF (via pypdf), plain text (.txt), Markdown (.md), and Word documents (.docx via python-docx)
- **Smart chunking** — word-boundary aware, configurable chunk size and overlap
- **Local embeddings** — `all-MiniLM-L6-v2` (22 M params, 384-dim, runs on CPU in milliseconds)
- **FAISS vector index** — exact flat L2 search, persisted to disk across restarts
- **Relevance scoring** — L2 distance converted to a [0, 1] similarity score
- **REST API** — clean FastAPI backend with Pydantic-validated schemas and OpenAPI docs
- **React UI** — dark-themed, responsive split-panel interface with upload, search, and document management
- **Docker Compose** — one command to start the full stack
- **CI/CD** — GitHub Actions pipeline covering both Python and TypeScript

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 |
| Vector index | `faiss-cpu` — FlatL2 |
| Document parsing | `pypdf` (PDF), `python-docx` (DOCX) |
| Backend | `FastAPI` + `uvicorn` |
| Data validation | `Pydantic v2` |
| Frontend | `React 18` + `TypeScript 5` + `Vite` |
| Styling | `Tailwind CSS v3` |
| Containerisation | `Docker` + `Docker Compose` |
| Web server | `nginx` (production frontend) |
| CI | `GitHub Actions` |

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/Shivansh2904/vector-vault.git
cd vector-vault
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

> **Note:** The first build downloads the `all-MiniLM-L6-v2` model (~90 MB). Subsequent starts use the cached layer.

### Option B — Manual setup

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend** (in a separate terminal):
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 — the dev server proxies API calls to port 8000.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload a document (multipart/form-data, field: `file`). Returns `doc_id`, `filename`, `chunk_count`. |
| `GET` | `/documents` | List all indexed documents with metadata. |
| `GET` | `/documents/{doc_id}` | Get metadata for a single document (`doc_id`, `filename`, `chunk_count`, `created_at`). Returns 404 if not found. |
| `GET` | `/documents/{doc_id}/chunks` | Inspect the text chunks a document was split into. |
| `DELETE` | `/documents/{doc_id}` | Remove a document and all its vectors from the index. |
| `POST` | `/search` | Search the corpus. Body: `{"query": "...", "top_k": 5, "rerank": false, "candidates_multiplier": 3}`. When `rerank` is true, fetches `top_k * candidates_multiplier` candidates via the bi-encoder and re-scores them with a cross-encoder for higher accuracy. Returns ranked chunks with scores (and `rerank_score` when reranking). |
| `GET` | `/health` | API status, total documents, total chunks, index vector count. |

Full interactive docs available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

**Example — upload:**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@report.pdf"
# {"doc_id":"3fa85f64-...","filename":"report.pdf","chunk_count":42}
```

**Example — search:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what are the key risks?", "top_k": 3}'
```

**Example — search with cross-encoder reranking:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what are the key risks?", "top_k": 3, "rerank": true, "candidates_multiplier": 4}'
```

---

## Two-Stage Retrieval (Bi-encoder + Cross-encoder)

VectorVault supports the standard production search pattern of **two-stage retrieval**:

1. **Stage 1 — Recall (bi-encoder):** the FAISS index returns a large candidate pool (`top_k * candidates_multiplier`) using cosine similarity on independently-embedded query and chunk vectors. This is fast — sub-millisecond per query against tens of thousands of vectors — but the bi-encoder never sees the query and chunk together, so its ranking is coarse.
2. **Stage 2 — Precision (cross-encoder):** the candidate pool is re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) which takes `(query, chunk)` as a single input and outputs a relevance score. Cross-encoders are slower (no pre-computed vectors — every query-doc pair runs through the model) but materially more accurate.

Enable it per-request by sending `"rerank": true`. The cross-encoder model is downloaded lazily on first use (~90 MB). When reranking is on, each result also includes a `rerank_score` alongside the original bi-encoder `score`.

```
Query  ─▶  Bi-encoder  ─▶  FAISS  ─▶  N candidates  ─▶  Cross-encoder  ─▶  Top-K
            (fast)                     (e.g. 15)         (accurate)         (e.g. 5)
```

**When to use it:** higher recall queries where the top result really matters (chatbot grounding, RAG, "find the one paragraph that answers this"). The latency cost is roughly 50–200 ms for 15–30 candidates on CPU.

---

## Project Structure

```
vector-vault/
├── backend/
│   ├── main.py           # FastAPI app, routes, request/response schemas
│   ├── chunker.py        # Word-boundary text chunking utility
│   ├── embedder.py       # sentence-transformers wrapper (L2-normalised)
│   ├── reranker.py       # Cross-encoder reranker for two-stage retrieval
│   ├── store.py          # FAISS index + metadata store with persistence
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx       # Main React application (split-panel UI)
│   │   ├── api.ts        # Typed API client
│   │   ├── main.tsx      # React entry point
│   │   └── index.css     # Tailwind base + custom utilities
│   ├── index.html
│   ├── nginx.conf        # SPA-routing nginx config
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── Dockerfile        # Multi-stage Node build → nginx serve
│
├── data/                 # FAISS index + metadata (git-ignored, Docker volume)
│   ├── index.faiss
│   └── metadata.json
│
├── .github/
│   └── workflows/
│       └── ci.yml        # Parallel Python + TypeScript CI jobs
│
├── docker-compose.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## Performance

| Operation | Typical latency | Notes |
|---|---|---|
| Embedding (single chunk) | ~5–15 ms | CPU, all-MiniLM-L6-v2 |
| Embedding (batch, 32 chunks) | ~80–150 ms | CPU |
| FAISS search (10K vectors) | < 1 ms | FlatL2, exact |
| FAISS search (1M vectors) | ~20–50 ms | FlatL2, exact |
| Document upload (1-page PDF) | ~200–400 ms | Including embedding |
| Document upload (100-page PDF) | ~3–8 s | Including embedding |

**Scaling notes:**
- For corpora over ~500K vectors, consider switching to `IndexIVFFlat` (approximate) or `IndexHNSW` for sub-millisecond search.
- Embedding throughput scales linearly with CPU cores; a GPU would give ~10–50× speedup via `faiss-gpu`.
- The `FlatL2` index memory footprint is `n_vectors × 384 × 4 bytes` — 1M vectors ≈ 1.5 GB RAM.

---

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/` | Directory for FAISS index and metadata JSON |
| `VITE_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |

---

## Development

**Run tests / linting:**
```bash
# Backend — syntax check all files
cd backend && python -m py_compile main.py chunker.py embedder.py store.py

# Frontend — type check + build
cd frontend && npm run build
```

**API docs** are auto-generated at `/docs` and `/redoc` by FastAPI.

---

## License

MIT © 2024 [Shivansh Mishra](https://github.com/Shivansh2904)
