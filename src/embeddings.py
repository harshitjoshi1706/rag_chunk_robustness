"""Normalized MiniLM embeddings for chunks and individual queries.

This module only encodes text. It does not build indexes or evaluate retrieval.
"""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32


def load_embedding_model(device: str = "cpu") -> SentenceTransformer:
    """Load the fixed experiment model; CPU is the reproducible default."""
    model = SentenceTransformer(MODEL_NAME, device=device)
    if model.get_embedding_dimension() != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings")
    model.eval()
    return model


def _encode(texts: Sequence[str], model: SentenceTransformer, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(texts, str) or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must be a sequence of nonempty strings")
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    expected_shape = (len(texts), EMBEDDING_DIMENSION)
    if vectors.shape != expected_shape:
        raise ValueError(f"Expected embedding shape {expected_shape}, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    return vectors


def embed_chunk_texts(
    texts: Sequence[str], model: SentenceTransformer, batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """Return a float32 matrix of unit-normalized chunk vectors, shape (N, 384)."""
    return _encode(texts, model, batch_size)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Return one unit-normalized float32 query vector, shape (384,)."""
    return _encode([query], model, DEFAULT_BATCH_SIZE)[0]
