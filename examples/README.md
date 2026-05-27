# Examples

## bulk_index_and_search.py

Indexes every PDF/TXT/MD/DOCX in a directory, then runs a semantic search query against the result.

```bash
# Terminal 1: start the backend
cd backend && uvicorn main:app --reload

# Terminal 2
pip install httpx

# Basic (bi-encoder only)
python examples/bulk_index_and_search.py ./my-docs "deep learning fundamentals"

# With cross-encoder reranking for higher accuracy
python examples/bulk_index_and_search.py ./my-docs "deep learning fundamentals" --rerank --top-k 10
```

Example output:

```
Indexing 8 files from /Users/you/my-docs...
  [ 12 chunks] paper-on-transformers.pdf
  [  4 chunks] notes.md
  [ 18 chunks] deeplearning-book-ch3.pdf
  ...

Searching for: 'deep learning fundamentals' (top_k=5, rerank=False)
  1. deeplearning-book-ch3.pdf      score=0.7421
     Deep learning is a subset of machine learning that uses multi-layer...
  2. paper-on-transformers.pdf      score=0.6892
     The Transformer architecture relies entirely on attention mechanisms...
  ...
```

## Tip

Pass `--rerank` for higher accuracy on ambiguous queries — the bi-encoder retrieves `top_k * candidates_multiplier` initial hits and a cross-encoder rescores them. Trade-off: ~5-10x slower per query.
