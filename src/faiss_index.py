"""Flat inner-product indexing for normalized 384-dimensional embeddings.

Returned integer positions follow insertion order; callers must keep their own
position-to-chunk metadata. No corpus loading or ground-truth logic lives here.
"""

from pathlib import Path

import faiss
import numpy as np

from embeddings import EMBEDDING_DIMENSION


def create_index() -> faiss.IndexFlatIP:
    """Create an exact-search cosine-equivalent index for unit vectors."""
    return faiss.IndexFlatIP(EMBEDDING_DIMENSION)


def validate_embedding_dimensions(vectors: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 matrix, or raise for invalid vectors."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected embedding matrix with shape (N, {EMBEDDING_DIMENSION}); got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must be finite")
    if matrix.shape[0] and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise ValueError("Embedding vectors must have unit L2 norm")
    return np.ascontiguousarray(matrix)


def _validate_index(index: faiss.Index) -> None:
    if not isinstance(index, faiss.IndexFlatIP) or index.d != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected a {EMBEDDING_DIMENSION}-dimensional FAISS IndexFlatIP")


def add_vectors(index: faiss.IndexFlatIP, vectors: np.ndarray) -> int:
    """Append vectors; return count added. An empty matrix is a no-op."""
    _validate_index(index)
    matrix = validate_embedding_dimensions(vectors)
    if matrix.shape[0]:
        index.add(matrix)
    return matrix.shape[0]


def save_index(index: faiss.IndexFlatIP, path: str | Path) -> None:
    """Save the index; this does not save position-to-chunk metadata."""
    _validate_index(index)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(destination))


def load_index(path: str | Path) -> faiss.IndexFlatIP:
    """Load and validate an IndexFlatIP from disk."""
    index = faiss.read_index(str(path))
    _validate_index(index)
    return index


def search_index(
    index: faiss.IndexFlatIP, queries: np.ndarray, k: int, num_threads: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inner-product scores, integer positions), each shape (N, k)."""
    _validate_index(index)
    if k <= 0:
        raise ValueError("k must be positive")
    if num_threads <= 0:
        raise ValueError("num_threads must be positive")
    matrix = np.asarray(queries, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    matrix = validate_embedding_dimensions(matrix)
    if index.ntotal == 0:
        raise ValueError("Cannot search an empty index")
    # FAISS/OpenMP can oversubscribe small Apple Silicon machines. This sets
    # FAISS's process-wide thread count; callers can opt into more threads.
    faiss.omp_set_num_threads(num_threads)
    return index.search(matrix, k)
