"""
chunker.py — Word-boundary-aware text chunking utility for VectorVault.
"""

from __future__ import annotations

import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* words,
    with *overlap* words of context carried over between consecutive chunks.

    Parameters
    ----------
    text:
        Raw document text (any length).
    chunk_size:
        Target number of words per chunk.
    overlap:
        Number of words from the end of one chunk that are repeated at the
        start of the next chunk.  Must be less than chunk_size.

    Returns
    -------
    List[str]
        Non-empty, whitespace-stripped chunk strings.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    # Normalise whitespace — collapse runs of spaces/tabs/newlines into a
    # single space so word-splitting is deterministic.
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    words: List[str] = text.split(" ")
    chunks: List[str] = []

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        # Advance by (chunk_size - overlap) so the next chunk starts
        # overlap words before the current chunk ended.
        start += chunk_size - overlap

    return chunks
