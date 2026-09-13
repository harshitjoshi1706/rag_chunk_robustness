from typing import Dict, List

import numpy as np


def summarize_chunk_lengths(
    chunks: List[Dict],
) -> Dict:
    """
    Summarize token-length characteristics of a chunk collection.

    These statistics help detect unfair differences in chunk size
    between chunking strategies.
    """

    if not chunks:
        raise ValueError(
            "Cannot summarize an empty chunk list."
        )

    token_lengths = np.array(
        [
            chunk["token_count"]
            for chunk in chunks
        ],
        dtype=np.float32,
    )

    return {
        "num_chunks": len(chunks),
        "mean_tokens": float(
            token_lengths.mean()
        ),
        "median_tokens": float(
            np.median(token_lengths)
        ),
        "std_tokens": float(
            token_lengths.std()
        ),
        "min_tokens": int(
            token_lengths.min()
        ),
        "max_tokens": int(
            token_lengths.max()
        ),
    }