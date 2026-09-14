"""Deterministic OHRBench document assembly with stable page provenance."""

from pathlib import Path

import pandas as pd


CORPUS_PATH = Path("data/raw/OHR-Bench_v2.parquet")
TEXT_COLUMNS = {
    "clean": "gt_text",
    "mild": "formatting_noise_mild",
    "moderate": "formatting_noise_moderate",
    "severe": "formatting_noise_severe",
}


def page_id(page_idx: int) -> str:
    return f"PAGE_{int(page_idx)}"


def load_pages(noise_level: str, path: Path = CORPUS_PATH) -> pd.DataFrame:
    if noise_level not in TEXT_COLUMNS:
        raise ValueError(f"Unknown formatting condition: {noise_level}")
    column = TEXT_COLUMNS[noise_level]
    pages = pd.read_parquet(path, columns=["domain", "doc_name", "page_idx", column])
    if pages[["domain", "doc_name", "page_idx", column]].isna().any().any():
        raise ValueError("OHRBench page fields contain null values")
    if pages.duplicated(["doc_name", "page_idx"]).any():
        raise ValueError("Duplicate document/page identity")
    return pages.sort_values(["doc_name", "page_idx"]).reset_index(drop=True)


def iter_documents(pages: pd.DataFrame, noise_level: str):
    """Yield one document at a time, retaining every original page index."""
    column = TEXT_COLUMNS[noise_level]
    for doc_name, group in pages.groupby("doc_name", sort=True):
        yield {
            "document_id": doc_name,
            "title": doc_name,
            "domain": group["domain"].iloc[0],
            "noise_level": noise_level,
            "paragraphs": [
                {"paragraph_id": page_id(row.page_idx), "page_idx": int(row.page_idx),
                 "text": getattr(row, column)}
                for row in group.itertuples(index=False)
            ],
        }


def attach_page_provenance(chunks: list[dict]) -> list[dict]:
    """Make source-page provenance explicit while retaining paragraph IDs."""
    for chunk in chunks:
        ids = chunk["source_paragraph_ids"]
        if any(not value.startswith("PAGE_") for value in ids):
            raise ValueError("Chunk has a non-page source ID")
        chunk["source_page_ids"] = sorted({int(value[5:]) for value in ids})
        if not chunk["source_page_ids"]:
            raise ValueError("Chunk has no source-page provenance")
    return chunks
