/**
 * api.ts — Typed API client for the VectorVault FastAPI backend.
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface UploadResponse {
  doc_id: string
  filename: string
  chunk_count: number
}

export interface Document {
  doc_id: string
  filename: string
  chunk_count: number
  created_at: string
}

export interface SearchResult {
  doc_id: string
  doc_filename: string
  chunk_text: string
  score: number
}

export interface SearchResponse {
  query: string
  results: SearchResult[]
}

export interface HealthResponse {
  status: string
  total_documents: number
  total_chunks: number
  index_vectors: number
  embedding_model: string
  embedding_dimension: number
}

export interface DeleteResponse {
  doc_id: string
  deleted: boolean
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const body = await res.json()
      message = body?.detail ?? JSON.stringify(body)
    } catch {
      // ignore parse errors
    }
    throw new Error(message)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

/**
 * Upload a document file (PDF, TXT, or MD) and index it in FAISS.
 */
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${BASE_URL}/documents/upload`, {
    method: 'POST',
    body: form,
  })
  return handleResponse<UploadResponse>(res)
}

/**
 * Fetch the list of all indexed documents.
 */
export async function listDocuments(): Promise<Document[]> {
  const res = await fetch(`${BASE_URL}/documents`)
  return handleResponse<Document[]>(res)
}

/**
 * Delete a document and all its vectors from the index.
 */
export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
  await handleResponse<DeleteResponse>(res)
}

/**
 * Run a semantic search query against the indexed corpus.
 */
export async function search(query: string, topK: number): Promise<SearchResult[]> {
  const res = await fetch(`${BASE_URL}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  })
  const data = await handleResponse<SearchResponse>(res)
  return data.results
}

/**
 * Ping the backend health endpoint.
 */
export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`)
  return handleResponse<HealthResponse>(res)
}
