"""Load the fixed official OHRBench page-level evaluation set."""

import json
from pathlib import Path

from qa_mapping_audit import normalize


QA_PATH = Path("data/raw/qas_v2.json")


def load_mapped_qas(clean_pages, path: Path = QA_PATH):
    """Return (mapped QAs, exclusions) without using answers or retrieval output."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(raw) != 8498 or len({row["ID"] for row in raw}) != 8498:
        raise ValueError("Official QA count or unique ID check failed")
    page_text = {(row.doc_name, int(row.page_idx)): row.gt_text
                 for row in clean_pages.itertuples(index=False)}
    documents = {row.doc_name for row in clean_pages.itertuples(index=False)}
    normalized_pages = {}
    mapped = []
    excluded = []
    for row in raw:
        doc_name = row["doc_name"]
        page_numbers = row["evidence_page_no"]
        page_numbers = page_numbers if isinstance(page_numbers, list) else [page_numbers]
        evidence_pages = tuple(sorted(set(page_numbers)))
        if doc_name not in documents:
            excluded.append({"query_id": row["ID"], "doc_name": doc_name,
                             "reason": "document_absent_from_local_corpus"})
            continue
        if not evidence_pages or any((doc_name, page) not in page_text for page in evidence_pages):
            raise ValueError(f"Evidence page mapping failed for QA {row['ID']}")
        contexts = row["evidence_context"]
        if isinstance(row["evidence_page_no"], list):
            if not isinstance(contexts, list) or len(contexts) != len(page_numbers):
                raise ValueError(f"Evidence pairing failed for QA {row['ID']}")
            pairs = zip(page_numbers, contexts)
        else:
            pairs = ((page_numbers[0], text) for text in
                     (contexts if isinstance(contexts, list) else [contexts]))
        verified = True
        for page, context in pairs:
            source = page_text[(doc_name, page)]
            if context in source:
                continue
            key = (doc_name, page)
            if key not in normalized_pages:
                normalized_pages[key] = normalize(source)
            if normalize(context) not in normalized_pages[key]:
                verified = False
        mapped.append({"query_id": row["ID"], "question": row["questions"],
                       "doc_name": doc_name, "domain": doc_name.split("/", 1)[0],
                       "evidence_pages": evidence_pages, "context_verified": verified})
    if len(mapped) != 8456 or len(excluded) != 42 or sum(q["context_verified"] for q in mapped) != 5277:
        raise ValueError("Phase 7B mapping counts did not reproduce")
    return mapped, excluded
