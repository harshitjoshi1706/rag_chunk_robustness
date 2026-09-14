"""Page-level relevance and query metrics for top-100 rankings."""


def is_page_relevant(chunk: dict, qa: dict) -> bool:
    return (chunk["document_id"] == qa["doc_name"]
            and bool(set(chunk["source_page_ids"]) & set(qa["evidence_pages"])))


def score_ranking(ranked_chunks: list[dict], qa: dict) -> dict:
    """Score a searched ranking; reciprocal rank is truncated at its depth."""
    first = next((rank for rank, chunk in enumerate(ranked_chunks, 1)
                  if is_page_relevant(chunk, qa)), None)
    return {
        "hit_at_1": int(first == 1),
        "hit_at_5": int(first is not None and first <= 5),
        "reciprocal_rank": 1.0 / first if first is not None else 0.0,
        "first_relevant_rank": first,
    }
