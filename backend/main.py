"""
main.py — VectorVault FastAPI application.

Endpoints
---------
POST   /documents/upload         Upload and index a document (PDF, TXT, MD)
GET    /documents                List all indexed documents
GET    /documents/{doc_id}       Get a single document's metadata
GET    /documents/{doc_id}/chunks Inspect chunks for a single document
DELETE /documents/{doc_id}       Remove a document from the index
POST   /search                   Semantic search over the indexed corpus
GET    /health                   API health check with index statistics
"""

from __future__ import annotations

import io
import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chunker import chunk_text
from embedder import Embedder
from store import DocumentStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("vectorvault")

# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VectorVault",
    description="Local-first semantic document search powered by FAISS + sentence-transformers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons (loaded once at startup)
# ---------------------------------------------------------------------------

embedder = Embedder()
store = DocumentStore(dimension=embedder.dimension)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    created_at: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class SearchResult(BaseModel):
    doc_id: str
    doc_filename: str
    chunk_text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    index_vectors: int
    embedding_model: str
    embedding_dimension: int


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool


class ChunkInfo(BaseModel):
    chunk_index: int
    text: str


class DocumentChunksResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: list[ChunkInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from uploaded file bytes."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == ".pdf":
        try:
            import pypdf  # lazy import; only needed for PDFs
        except ImportError as exc:
            raise HTTPException(
                status_code=500, detail="pypdf not installed — cannot process PDFs"
            ) from exc

        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if ext == ".docx":
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="python-docx not installed") from exc
        doc = DocxDocument(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if ext in {".txt", ".md", ""}:
        # Try common encodings gracefully.
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=400, detail="Could not decode file as text")

    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    summary="Upload and index a document",
    tags=["documents"],
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a PDF, TXT, or MD file, chunk it, embed it, and store it in FAISS."""
    filename = file.filename or "unnamed"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    log.info("Received file: %s (%d bytes)", filename, len(data))

    # Extract raw text
    text = _extract_text(filename, data)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in the document")

    # Chunk
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no text chunks")

    log.info("Chunked into %d chunks", len(chunks))

    # Embed
    embeddings = embedder.embed(chunks)

    # Store
    doc_id = store.add_document(filename=filename, chunks=chunks, embeddings=embeddings)
    log.info("Indexed document %s with %d chunks", doc_id, len(chunks))

    return UploadResponse(doc_id=doc_id, filename=filename, chunk_count=len(chunks))


@app.get(
    "/documents",
    response_model=list[DocumentMeta],
    summary="List all indexed documents",
    tags=["documents"],
)
async def list_documents() -> list[DocumentMeta]:
    """Return metadata for every document currently in the index."""
    docs = store.list_documents()
    return [DocumentMeta(**d) for d in docs]


@app.get(
    "/documents/{doc_id}",
    response_model=DocumentMeta,
    summary="Get a single document's metadata",
    tags=["documents"],
)
async def get_document(doc_id: str) -> DocumentMeta:
    """Return metadata for a single document by its doc_id."""
    docs = store.list_documents()
    for doc in docs:
        if doc["doc_id"] == doc_id:
            return DocumentMeta(**doc)
    raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")


@app.get(
    "/documents/{doc_id}/chunks",
    response_model=DocumentChunksResponse,
    summary="Get all chunks for a document",
    tags=["documents"],
)
async def get_document_chunks(doc_id: str) -> DocumentChunksResponse:
    """Return all text chunks for a given document, useful for inspecting how it was split."""
    try:
        chunks = store.get_chunks(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

    # Get filename from metadata
    docs = store.list_documents()
    doc_meta = next((d for d in docs if d["doc_id"] == doc_id), None)
    filename = doc_meta["filename"] if doc_meta else "unknown"

    return DocumentChunksResponse(
        doc_id=doc_id,
        filename=filename,
        chunks=[ChunkInfo(**c) for c in chunks],
    )


@app.delete(
    "/documents/{doc_id}",
    response_model=DeleteResponse,
    summary="Remove a document from the index",
    tags=["documents"],
)
async def delete_document(doc_id: str) -> DeleteResponse:
    """Delete a document and all its associated vectors."""
    try:
        store.remove_document(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    log.info("Deleted document %s", doc_id)
    return DeleteResponse(doc_id=doc_id, deleted=True)


@app.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search",
    tags=["search"],
)
async def search(request: SearchRequest) -> SearchResponse:
    """Embed the query and return the top-K most similar document chunks."""
    stats = store.get_stats()
    if stats["total_chunks"] == 0:
        return SearchResponse(query=request.query, results=[])

    query_vec = embedder.embed_query(request.query)
    raw = store.search(query_vec, top_k=request.top_k)

    results = [SearchResult(**r) for r in raw]
    return SearchResponse(query=request.query, results=results)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["system"],
)
async def health() -> HealthResponse:
    """Return API status and current index statistics."""
    stats = store.get_stats()
    return HealthResponse(
        status="ok",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=embedder.dimension,
        **stats,
    )
