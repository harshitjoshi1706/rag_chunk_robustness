"""Clean-corpus chunk-size diagnostics for Phase 4E (never retrieval tuning)."""

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from chunkers import MODEL_NAME, semantic_chunk, validate_semantic_chunks
from profiling import summarize_chunk_lengths

DATASET = Path("data/raw/OHR-Bench_v2.parquet")
THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
PAGES_PER_DOMAIN = 2


def select_clean_sample():
    """Deterministic, distinct-document pages of typical length per domain."""
    df = pd.read_parquet(DATASET, columns=["domain", "doc_name", "page_idx", "gt_text"])
    df = df.dropna(subset=["domain", "doc_name", "page_idx", "gt_text"]).copy()
    df["characters"] = df["gt_text"].str.len()
    df = df[df["characters"].between(1000, 5000)]
    selected = []
    for domain, group in df.groupby("domain", sort=True):
        median = group["characters"].median()
        group = group.assign(distance=(group["characters"] - median).abs())
        group = group.sort_values(["distance", "doc_name", "page_idx"])
        pages = group.drop_duplicates("doc_name").head(PAGES_PER_DOMAIN)
        if len(pages) != PAGES_PER_DOMAIN:
            raise ValueError(f"Insufficient distinct documents in {domain}")
        selected.extend(pages.to_dict("records"))
    if len(selected) != 7 * PAGES_PER_DOMAIN:
        raise ValueError("Expected clean pages from all seven domains")
    return selected


class CachedEncoder:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def encode(self, texts, **kwargs):
        key = tuple(texts)
        if key not in self.cache:
            self.cache[key] = self.model.encode(texts, **kwargs)
        return self.cache[key]


def main():
    sample = select_clean_sample()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CachedEncoder(SentenceTransformer(MODEL_NAME))
    print("CLEAN diagnostic sample (not research results):")
    for row in sample:
        print(f"  {row['domain']} | {row['doc_name']} | page {row['page_idx']} | {row['characters']} chars")
    print("threshold chunks mean median std min max short(<64) budget(>=240)")
    for threshold in THRESHOLDS:
        all_chunks = []
        for row in sample:
            document = {
                "document_id": f"{row['domain']}:{row['doc_name']}:{row['page_idx']}",
                "paragraphs": [{"paragraph_id": f"page:{row['page_idx']}", "text": row["gt_text"]}],
            }
            chunks = semantic_chunk(document, tokenizer, model, similarity_threshold=threshold,
                                    min_tokens=64, max_tokens=256)
            if not validate_semantic_chunks(chunks, 256):
                raise AssertionError(f"Invalid chunks for {document['document_id']}")
            all_chunks.extend(chunks)
        stats = summarize_chunk_lengths(all_chunks)
        short = sum(c["token_count"] < 64 for c in all_chunks)
        budget = sum(c["token_count"] >= 240 for c in all_chunks)
        print(f"{threshold:.2f} {stats['num_chunks']} {stats['mean_tokens']:.2f} "
              f"{stats['median_tokens']:.2f} {stats['std_tokens']:.2f} "
              f"{stats['min_tokens']} {stats['max_tokens']} {short} {budget}")


if __name__ == "__main__":
    main()
