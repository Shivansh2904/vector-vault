"""Chunking tests.

The bi-encoder (all-MiniLM-L6-v2) reads at most 256 tokens per input and drops
the rest without an error. Chunks longer than that are only partly searchable,
so these tests check chunk sizes in the model's own tokens and check that text
deep inside a document can still be found.
"""

from __future__ import annotations

import pytest

from chunker import chunk_text

# Plain prose with nothing in common with the needle below.
FILLER_SENTENCES = [
    "The allotment committee met on Saturday morning to plan the spring planting.",
    "Most plots along the east fence will grow potatoes, onions and runner beans this year.",
    "A new water butt was fitted beside the shed after the old one cracked in the frost.",
    "Members agreed to share the rotavator on a weekly rota posted by the gate.",
    "Compost from the three bays at the back is ready to spread once the soil warms up.",
    "Slugs damaged most of the lettuce seedlings last April, so beer traps will go in early.",
    "The raspberry canes were cut back to knee height and tied to fresh wire.",
    "Children from the primary school will sow sunflowers in the shared bed in May.",
    "Anyone leaving a plot untended for two months will receive a friendly reminder letter.",
    "The treasurer reported that seed orders came in slightly under budget.",
    "Wood chip for the paths is delivered free by the council tree surgeons.",
    "Rhubarb forcing pots are available to borrow from the tool store.",
]

NEEDLE = "The spare key to the observatory telescope dome is taped under the third library bench."
NEEDLE_QUERY = "where is the spare key for the observatory telescope dome hidden"


def filler_words(n: int) -> list[str]:
    words: list[str] = []
    while len(words) < n:
        for sentence in FILLER_SENTENCES:
            words.extend(sentence.split())
    return words[:n]


def model_of(app_module):
    """The SentenceTransformer behind the app's embedder."""
    return app_module.embedder._model


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------


def test_every_chunk_fits_the_embedding_window(client, upload, app_module):
    model = model_of(app_module)
    doc_id = upload("long.txt", " ".join(filler_words(1500)))["doc_id"]

    chunks = client.get(f"/documents/{doc_id}/chunks").json()["chunks"]
    assert len(chunks) > 1
    for c in chunks:
        n_tokens = len(model.tokenizer(c["text"])["input_ids"])  # includes [CLS]/[SEP]
        assert n_tokens <= model.max_seq_length, (
            f"chunk {c['chunk_index']} has {n_tokens} tokens; the model reads "
            f"{model.max_seq_length} and silently drops the rest"
        )


@pytest.mark.parametrize("needle_at_word", [20, 250, 400])
def test_needle_is_found_wherever_it_sits(client, upload, needle_at_word):
    # Three distractors hold exactly the same filler without the needle, so the
    # only thing that can rank the needle document above them is the needle
    # being inside a window the model actually embedded.
    filler = filler_words(600)
    words = filler[:needle_at_word] + NEEDLE.split() + filler[needle_at_word:]
    for i in range(3):
        upload(f"distractor{i}.txt", " ".join(filler))
    needle_doc = upload("needle.txt", " ".join(words))["doc_id"]

    r = client.post("/search", json={"query": NEEDLE_QUERY, "top_k": 20})
    results = r.json()["results"]

    top = results[0]
    best_distractor = max(x["score"] for x in results if x["doc_id"] != needle_doc)
    assert top["doc_id"] == needle_doc
    assert NEEDLE in top["chunk_text"]
    assert top["score"] > best_distractor + 0.05, (top["score"], best_distractor)


# ---------------------------------------------------------------------------
# chunk_text itself
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer(app_module):
    return app_module.embedder.tokenizer


def n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_chunks_cover_the_text_in_order_with_overlap(tokenizer):
    text = " ".join(filler_words(1200))
    chunks = chunk_text(text, tokenizer, max_tokens=200, overlap=40)
    assert len(chunks) > 3

    assert text.startswith(chunks[0])
    assert text.endswith(chunks[-1])
    prev_start, prev_end = 0, len(chunks[0])
    for chunk in chunks[1:]:
        start = text.find(chunk, prev_start + 1)
        assert start != -1
        assert prev_start < start < prev_end, "next chunk must start inside the previous one"
        prev_start, prev_end = start, start + len(chunk)
    assert prev_end == len(text)


def test_chunks_stay_under_max_tokens_and_end_between_words(tokenizer):
    text = " ".join(filler_words(1200))
    words = set(text.split())
    for chunk in chunk_text(text, tokenizer, max_tokens=200, overlap=40):
        assert n_tokens(tokenizer, chunk) <= 200
        assert chunk.split()[0] in words and chunk.split()[-1] in words


def test_whitespace_is_collapsed_and_case_kept(tokenizer):
    assert chunk_text("  Hello\n\n  World\t! ", tokenizer) == ["Hello World !"]


def test_a_word_longer_than_a_chunk_is_split(tokenizer):
    # No spaces, but the tokenizer splits on every "." (about 400 tokens).
    blob = ".".join(["word"] * 200)
    chunks = chunk_text(f"start {blob} end", tokenizer, max_tokens=50, overlap=10)
    assert len(chunks) > 1
    assert all(n_tokens(tokenizer, c) <= 50 for c in chunks)


def test_blank_text_gives_no_chunks(tokenizer):
    assert chunk_text(" \n\t ", tokenizer) == []


def test_overlap_must_be_smaller_than_the_chunk(tokenizer):
    with pytest.raises(ValueError):
        chunk_text("some text", tokenizer, max_tokens=40, overlap=40)
