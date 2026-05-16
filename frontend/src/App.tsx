import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  deleteDocument,
  healthCheck,
  listDocuments,
  search,
  uploadDocument,
  type Document,
  type HealthResponse,
  type SearchResult,
} from './api'

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function HealthBadge({ health }: { health: HealthResponse | null; error: boolean }) {
  if (!health) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-slate-500">
        <span className="w-2 h-2 rounded-full bg-slate-600 animate-pulse-slow" />
        connecting…
      </div>
    )
  }
  return (
    <div className="flex items-center gap-1.5 text-xs text-emerald-400">
      <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]" />
      API online · {health.total_documents} docs · {health.total_chunks} chunks
    </div>
  )
}

function SkeletonResult() {
  return (
    <div className="rounded-xl border border-border bg-surface-2 p-4 space-y-3">
      <div className="skeleton h-3 w-1/3 rounded" />
      <div className="skeleton h-2 w-full rounded" />
      <div className="skeleton h-2 w-5/6 rounded" />
      <div className="skeleton h-2 w-2/3 rounded" />
    </div>
  )
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color =
    pct >= 75 ? 'bg-emerald-500' : pct >= 50 ? 'bg-indigo-500' : pct >= 30 ? 'bg-amber-500' : 'bg-slate-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-surface-3 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-slate-400 w-10 text-right">{pct}%</span>
    </div>
  )
}

function ResultCard({ result, index }: { result: SearchResult; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const preview = result.chunk_text.slice(0, 280)
  const hasMore = result.chunk_text.length > 280

  return (
    <div className="rounded-xl border border-border bg-surface-2 p-4 space-y-3 hover:border-accent/40 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex-shrink-0 w-6 h-6 rounded-md bg-accent/20 text-accent text-xs font-bold flex items-center justify-center">
            {index + 1}
          </span>
          <span className="text-sm font-medium text-slate-200 truncate">{result.doc_filename}</span>
        </div>
        <span className="flex-shrink-0 text-xs text-slate-500 font-mono">
          {result.doc_id.slice(0, 8)}
        </span>
      </div>

      <ScoreBar score={result.score} />

      <p className="text-sm text-slate-300 leading-relaxed font-mono">
        {expanded ? result.chunk_text : preview}
        {hasMore && !expanded && '…'}
      </p>

      {hasMore && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-xs text-accent hover:text-accent-hover transition-colors"
        >
          {expanded ? 'Show less' : 'Show full chunk'}
        </button>
      )}
    </div>
  )
}

function DocumentRow({
  doc,
  onDelete,
  deleting,
}: {
  doc: Document
  onDelete: (id: string) => void
  deleting: boolean
}) {
  const date = new Date(doc.created_at).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })

  return (
    <div className="flex items-center justify-between gap-3 px-3 py-2.5 rounded-lg hover:bg-surface-3 transition-colors group">
      <div className="min-w-0 flex-1">
        <p className="text-sm text-slate-200 truncate" title={doc.filename}>
          {doc.filename}
        </p>
        <p className="text-xs text-slate-500 mt-0.5">
          {doc.chunk_count} chunks · {date}
        </p>
      </div>
      <button
        onClick={() => onDelete(doc.doc_id)}
        disabled={deleting}
        title="Delete document"
        className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-md hover:bg-red-900/40 hover:text-red-400 text-slate-500 disabled:opacity-30"
      >
        {deleting ? (
          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
          </svg>
        ) : (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        )}
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  // Health
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState(false)

  // Documents
  const [documents, setDocuments] = useState<Document[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set())
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Search
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [results, setResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [searched, setSearched] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)

  // ---------------------------------------------------------------------------
  // Data fetching
  // ---------------------------------------------------------------------------

  const fetchHealth = useCallback(async () => {
    try {
      const h = await healthCheck()
      setHealth(h)
      setHealthError(false)
    } catch {
      setHealthError(true)
    }
  }, [])

  const fetchDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments()
      setDocuments(docs)
    } catch {
      // silently fail — health badge shows connection status
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    fetchDocuments()
    const interval = setInterval(fetchHealth, 15_000)
    return () => clearInterval(interval)
  }, [fetchHealth, fetchDocuments])

  // ---------------------------------------------------------------------------
  // Upload
  // ---------------------------------------------------------------------------

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''

    setUploading(true)
    setUploadError(null)
    try {
      await uploadDocument(file)
      await Promise.all([fetchDocuments(), fetchHealth()])
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  // ---------------------------------------------------------------------------
  // Delete
  // ---------------------------------------------------------------------------

  const handleDelete = async (docId: string) => {
    setDeletingIds((s) => new Set([...s, docId]))
    try {
      await deleteDocument(docId)
      await Promise.all([fetchDocuments(), fetchHealth()])
    } catch (err) {
      console.error('Delete failed:', err)
    } finally {
      setDeletingIds((s) => {
        const next = new Set(s)
        next.delete(docId)
        return next
      })
    }
  }

  // ---------------------------------------------------------------------------
  // Search
  // ---------------------------------------------------------------------------

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setSearching(true)
    setSearchError(null)
    setSearched(false)
    try {
      const res = await search(query.trim(), topK)
      setResults(res)
      setSearched(true)
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setSearching(false)
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-border bg-surface-1/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Logo mark */}
            <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7C5 4 4 5 4 7z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6M9 16h4" />
              </svg>
            </div>
            <span className="font-semibold text-slate-100 tracking-tight">VectorVault</span>
            <span className="hidden sm:block text-xs text-slate-500 border border-border rounded px-2 py-0.5">
              local-first · no API keys
            </span>
          </div>
          <HealthBadge health={health} error={healthError} />
        </div>
      </header>

      {/* ── Main layout ── */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-6 flex flex-col lg:flex-row gap-6">

        {/* ══ Left panel — Document Library ══ */}
        <aside className="w-full lg:w-72 xl:w-80 flex-shrink-0 flex flex-col gap-4">
          <div className="rounded-xl border border-border bg-surface-1 p-4 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-200">Document Library</h2>
              <span className="text-xs text-slate-500 bg-surface-3 px-2 py-0.5 rounded-full">
                {documents.length}
              </span>
            </div>

            {/* Upload button */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt,.md"
              className="hidden"
              onChange={handleFileChange}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg border border-dashed border-border hover:border-accent/60 hover:bg-accent/5 transition-colors text-sm text-slate-400 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploading ? (
                <>
                  <svg className="w-4 h-4 animate-spin text-accent" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                  Indexing…
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Upload document
                </>
              )}
            </button>
            <p className="text-xs text-slate-600 text-center">PDF · TXT · Markdown</p>

            {uploadError && (
              <div className="text-xs text-red-400 bg-red-900/20 border border-red-900/40 rounded-lg px-3 py-2">
                {uploadError}
              </div>
            )}
          </div>

          {/* Document list */}
          <div className="rounded-xl border border-border bg-surface-1 flex-1 overflow-hidden">
            {documents.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 py-12 text-center px-6">
                <svg className="w-10 h-10 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm text-slate-500">No documents yet</p>
                <p className="text-xs text-slate-600">Upload a file to get started</p>
              </div>
            ) : (
              <div className="p-2 space-y-0.5 max-h-96 lg:max-h-[calc(100vh-22rem)] overflow-y-auto">
                {documents.map((doc) => (
                  <DocumentRow
                    key={doc.doc_id}
                    doc={doc}
                    onDelete={handleDelete}
                    deleting={deletingIds.has(doc.doc_id)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Index stats */}
          {health && (
            <div className="rounded-xl border border-border bg-surface-1 p-4 grid grid-cols-2 gap-3">
              {[
                { label: 'Documents', value: health.total_documents },
                { label: 'Chunks', value: health.total_chunks },
                { label: 'Vectors', value: health.index_vectors },
                { label: 'Dim', value: health.embedding_dimension },
              ].map(({ label, value }) => (
                <div key={label} className="bg-surface-2 rounded-lg px-3 py-2">
                  <p className="text-xs text-slate-500">{label}</p>
                  <p className="text-lg font-semibold text-slate-100 font-mono">{value}</p>
                </div>
              ))}
            </div>
          )}
        </aside>

        {/* ══ Right panel — Search ══ */}
        <section className="flex-1 flex flex-col gap-5 min-w-0">
          {/* Search form */}
          <div className="rounded-xl border border-border bg-surface-1 p-5 space-y-4">
            <form onSubmit={handleSearch} className="flex flex-col gap-3">
              <div className="relative">
                <svg
                  className="absolute left-3.5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500 pointer-events-none"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask anything about your documents…"
                  className="w-full pl-11 pr-4 py-3.5 bg-surface-2 border border-border rounded-xl text-slate-100 placeholder-slate-500 focus:outline-none focus:border-accent/60 focus:ring-1 focus:ring-accent/30 transition-colors text-sm"
                />
              </div>

              <div className="flex items-center gap-4">
                <div className="flex items-center gap-3 flex-1">
                  <label className="text-xs text-slate-500 whitespace-nowrap">
                    Top-K: <span className="text-accent font-semibold">{topK}</span>
                  </label>
                  <input
                    type="range"
                    min={1}
                    max={10}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="flex-1 accent-indigo-500 cursor-pointer"
                  />
                </div>

                <button
                  type="submit"
                  disabled={searching || !query.trim()}
                  className="flex items-center gap-2 px-5 py-2.5 bg-accent hover:bg-accent-hover disabled:bg-surface-3 disabled:text-slate-500 text-white text-sm font-medium rounded-xl transition-colors"
                >
                  {searching ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                      </svg>
                      Searching
                    </>
                  ) : (
                    'Search'
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Error */}
          {searchError && (
            <div className="text-sm text-red-400 bg-red-900/20 border border-red-900/40 rounded-xl px-4 py-3">
              {searchError}
            </div>
          )}

          {/* Skeleton */}
          {searching && (
            <div className="space-y-3">
              {Array.from({ length: topK > 3 ? 3 : topK }).map((_, i) => (
                <SkeletonResult key={i} />
              ))}
            </div>
          )}

          {/* Results */}
          {!searching && searched && (
            <div className="space-y-3">
              {results.length === 0 ? (
                <div className="rounded-xl border border-border bg-surface-1 p-10 text-center">
                  <p className="text-slate-400 text-sm">No matching chunks found.</p>
                  <p className="text-slate-600 text-xs mt-1">Try a different query or upload more documents.</p>
                </div>
              ) : (
                <>
                  <p className="text-xs text-slate-500 px-1">
                    {results.length} result{results.length !== 1 ? 's' : ''} for &ldquo;<span className="text-slate-300">{query}</span>&rdquo;
                  </p>
                  {results.map((r, i) => (
                    <ResultCard key={`${r.doc_id}-${i}`} result={r} index={i} />
                  ))}
                </>
              )}
            </div>
          )}

          {/* Empty state */}
          {!searching && !searched && !searchError && (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 py-20 text-center">
              <div className="w-16 h-16 rounded-2xl bg-surface-2 border border-border flex items-center justify-center">
                <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <div>
                <p className="text-slate-400 font-medium">Semantic search, locally</p>
                <p className="text-slate-600 text-sm mt-1 max-w-sm">
                  Upload documents and search them with natural language — embeddings run entirely on your machine.
                </p>
              </div>
              <div className="flex flex-wrap gap-2 justify-center">
                {['What are the main findings?', 'Summarise the key points', 'List all action items'].map((ex) => (
                  <button
                    key={ex}
                    onClick={() => setQuery(ex)}
                    className="text-xs px-3 py-1.5 rounded-full border border-border hover:border-accent/40 text-slate-400 hover:text-slate-200 transition-colors"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-border py-4 text-center text-xs text-slate-600">
        VectorVault · sentence-transformers + FAISS · no data leaves your machine
      </footer>
    </div>
  )
}
