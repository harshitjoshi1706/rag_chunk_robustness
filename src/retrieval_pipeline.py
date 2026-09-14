"""Chunk and search OHRBench documents using stable source-page metadata."""

from chunkers import (
    DEFAULT_FIXED_CHUNK_OVERLAP, DEFAULT_FIXED_CHUNK_SIZE,
    DEFAULT_SEMANTIC_MAX_TOKENS, DEFAULT_SEMANTIC_MIN_TOKENS,
    DEFAULT_SEMANTIC_THRESHOLD, DEFAULT_STRUCTURE_MAX_TOKENS,
    fixed_size_chunk, recursive_chunk, semantic_chunk, structure_aware_chunk,
)
from faiss_index import search_index
from ohrbench_adapter import attach_page_provenance


CHUNKERS = ("fixed", "recursive", "semantic", "structure")


def chunk_document(document, chunker, tokenizer, model):
    if chunker == "fixed":
        chunks = fixed_size_chunk(document, tokenizer, DEFAULT_FIXED_CHUNK_SIZE,
                                  DEFAULT_FIXED_CHUNK_OVERLAP)
    elif chunker == "recursive":
        chunks = recursive_chunk(document, tokenizer, DEFAULT_FIXED_CHUNK_SIZE,
                                 DEFAULT_FIXED_CHUNK_OVERLAP)
    elif chunker == "semantic":
        chunks = semantic_chunk(document, tokenizer, model, DEFAULT_SEMANTIC_THRESHOLD,
                                DEFAULT_SEMANTIC_MAX_TOKENS, DEFAULT_SEMANTIC_MIN_TOKENS)
    elif chunker == "structure":
        chunks = structure_aware_chunk(document, tokenizer, DEFAULT_STRUCTURE_MAX_TOKENS)
    else:
        raise ValueError(f"Unknown chunker: {chunker}")
    attach_page_provenance(chunks)
    for chunk in chunks:
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError(f"Empty retrievable chunk: {chunk['chunk_id']}")
        # Embedding model accepts 256 including [CLS] and [SEP].
        actual = len(tokenizer(chunk["text"], add_special_tokens=False)["input_ids"])
        if actual > 254:
            raise ValueError(f"Chunk would be truncated by embedding model: {chunk['chunk_id']} ({actual})")
        chunk["token_count"] = actual
        chunk["noise_level"] = document["noise_level"]
    return chunks


def retrieve_one(index, metadata, query_vector, k=100):
    """Return ordered (score, metadata) pairs; FAISS positions map by insertion order."""
    scores, positions = search_index(index, query_vector, k)
    return [(float(score), metadata[int(position)])
            for score, position in zip(scores[0], positions[0]) if position >= 0]
