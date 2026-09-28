# Examples

## bulk_index_and_search.py

Uploads every PDF, TXT, MD and DOCX file under a directory to a running
VectorVault API, then runs one search and prints the top results.

```bash
# Terminal 1: start the backend
cd backend && uvicorn main:app --reload

# Terminal 2, from the repository root
pip install httpx
python examples/bulk_index_and_search.py ./my-docs "deep learning fundamentals"

# Re-rank the candidates with the cross-encoder
python examples/bulk_index_and_search.py ./my-docs "deep learning fundamentals" --rerank --top-k 10
```

Each result shows the file name, the bi-encoder `score` and, with `--rerank`,
the cross-encoder's `rerank_score`, followed by the start of the matching
chunk.

The script does not skip files that are already indexed, so running it twice
on the same folder indexes everything twice. Set `VECTORVAULT_API` to point it
at an API other than `http://localhost:8000`.
