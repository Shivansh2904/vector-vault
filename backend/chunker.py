"""
chunker.py — Token-based text chunking for VectorVault.

Chunks are measured in the embedding model's own tokens, not in words. The
bi-encoder only reads its first ``max_seq_length`` tokens (256 for
all-MiniLM-L6-v2) and silently drops the rest, so a chunk that is longer than
that is only partly searchable.
"""

from __future__ import annotations

import re
from typing import Any


def chunk_text(
    text: str,
    tokenizer: Any,
    max_tokens: int = 200,
    overlap: int = 40,
) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *max_tokens* tokens.

    Parameters
    ----------
    text:
        Raw document text (any length). Runs of whitespace are collapsed to a
        single space first.
    tokenizer:
        A Hugging Face fast tokenizer (it must support
        ``return_offsets_mapping``), normally the embedding model's own.
    max_tokens:
        Maximum tokens per chunk, not counting the special tokens the model
        adds. Keep it at least 2 below the model's ``max_seq_length``.
    overlap:
        Roughly how many tokens consecutive chunks share, so a sentence that
        straddles a boundary still appears whole in one of them. Must be less
        than *max_tokens*.

    Returns
    -------
    list[str]
        Chunks sliced from the normalised text, so casing and punctuation are
        kept. A chunk ends between two words unless a single word is longer
        than *max_tokens*.
    """
    if overlap >= max_tokens:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than max_tokens ({max_tokens})"
        )

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,  # no "sequence longer than the model max" warning
    )
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []

    def starts_word(i: int) -> bool:
        # Token i starts a new word when there is a space between it and the
        # previous token.
        return i == 0 or offsets[i][0] > offsets[i - 1][1]

    chunks: list[str] = []
    start = 0
    while True:
        end = min(start + max_tokens, n)
        if end < n:
            # Back off so the word straddling the limit moves to the next chunk.
            cut = end
            while cut > start and not starts_word(cut):
                cut -= 1
            if cut > start:
                end = cut
        chunks.append(text[offsets[start][0] : offsets[end - 1][1]])
        if end == n:
            break

        next_start = end - overlap
        if next_start <= start:
            # A chunk no longer than the overlap: carry nothing over.
            next_start = end
        else:
            while next_start < end and not starts_word(next_start):
                next_start += 1
        start = next_start

    return chunks
