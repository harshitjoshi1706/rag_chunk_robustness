"""Phase 7B diagnostic: audit official QA annotations against clean OHRBench pages.

Read-only analysis; no relevance labels or retrieval results are generated.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd


QA_PATH = Path("data/raw/qas_v2.json")
CORPUS_PATH = Path("data/raw/OHR-Bench_v2.parquet")


def normalize(text):
    return " ".join(text.split())


def evidence_pairs(item):
    """Pair official page numbers with contexts, retaining multi-page evidence."""
    pages = item["evidence_page_no"]
    contexts = item["evidence_context"]
    if isinstance(pages, int):
        pages = [pages]
        contexts = contexts if isinstance(contexts, list) else [contexts]
        return [(pages[0], context) for context in contexts]
    if not isinstance(pages, list) or not isinstance(contexts, list) or len(pages) != len(contexts):
        raise ValueError(f"Unpaired evidence for QA {item['ID']}")
    return list(zip(pages, contexts))


def audit_qa():
    qas = json.loads(QA_PATH.read_text(encoding="utf-8"))
    corpus = pd.read_parquet(CORPUS_PATH, columns=["domain", "doc_name", "page_idx", "gt_text"])
    documents = set(corpus["doc_name"])
    pages = {(row.doc_name, row.page_idx): row.gt_text
             for row in corpus.itertuples(index=False)}

    result = {"total_qa": len(qas), "unique_qa_documents": len({q["doc_name"] for q in qas}),
              "local_documents": len(documents), "mapped_docs": 0, "unmapped_docs": 0,
              "doc_type_mismatches": 0, "duplicate_ids": len(qas) - len({q["ID"] for q in qas})}
    offsets = {str(offset): Counter() for offset in (0, -1, 1)}
    qa_by_domain = Counter()
    usable_by_domain = Counter()
    context_verified_by_domain = Counter()
    reason_counts = Counter()
    reference_counts = Counter()
    context_counts = Counter()
    qa_status = Counter()
    missing_examples = []
    ambiguous_examples = []
    normalized_cache = {}

    for qa in qas:
        doc = qa["doc_name"]
        if doc not in documents:
            result["unmapped_docs"] += 1
            reason_counts["unmapped_document"] += 1
            if len(missing_examples) < 5:
                missing_examples.append({"ID": qa["ID"], "doc_name": doc})
            continue
        result["mapped_docs"] += 1
        domain = doc.split("/", 1)[0]
        qa_by_domain[domain] += 1
        if qa["doc_type"] != domain:
            result["doc_type_mismatches"] += 1

        pairs = evidence_pairs(qa)
        reference_counts["total"] += len({page for page, _ in pairs})
        context_counts["total"] += len(pairs)
        all_exist = True
        all_exact = True
        all_normalized = True
        any_repeated = False
        for offset in (0, -1, 1):
            translated = [(doc, page + offset) for page in {page for page, _ in pairs}]
            if all(key in pages for key in translated):
                offsets[str(offset)]["qa_all_pages_exist"] += 1
            for key in translated:
                if key in pages:
                    offsets[str(offset)]["reference_exists"] += 1
                else:
                    offsets[str(offset)]["reference_missing"] += 1
            for page, context in pairs:
                key = (doc, page + offset)
                if key not in pages:
                    continue
                source = pages[key]
                if context and context in source:
                    offsets[str(offset)]["context_exact"] += 1
                else:
                    if key not in normalized_cache:
                        normalized_cache[key] = normalize(source)
                    if context and normalize(context) in normalized_cache[key]:
                        offsets[str(offset)]["context_normalized_only"] += 1

        for page, context in pairs:
            key = (doc, page)
            if key not in pages:
                all_exist = all_exact = all_normalized = False
                reason_counts["missing_page_reference"] += 1
                if len(missing_examples) < 5:
                    missing_examples.append({"ID": qa["ID"], "doc_name": doc, "page": page})
                continue
            source = pages[key]
            count = source.count(context) if context else 0
            if count:
                context_counts["exact"] += 1
                if count > 1:
                    context_counts["repeated_exact"] += 1
                    any_repeated = True
            else:
                all_exact = False
                if key not in normalized_cache:
                    normalized_cache[key] = normalize(source)
                normalized_count = normalized_cache[key].count(normalize(context)) if context else 0
                if normalized_count:
                    context_counts["normalized_only"] += 1
                    if normalized_count > 1:
                        context_counts["repeated_normalized_only"] += 1
                        any_repeated = True
                else:
                    context_counts["unresolved"] += 1
                    all_normalized = False
                    if len(ambiguous_examples) < 5:
                        ambiguous_examples.append({"ID": qa["ID"], "doc_name": doc,
                                                   "page": page, "context_prefix": context[:100]})
        if all_exist:
            qa_status["all_pages_exist"] += 1
            usable_by_domain[domain] += 1
        if all_exact:
            qa_status["all_contexts_exact"] += 1
        if all_normalized:
            qa_status["all_contexts_exact_or_normalized"] += 1
            context_verified_by_domain[domain] += 1
        if any_repeated:
            qa_status["repeated_context_on_annotated_page"] += 1

    result.update({"offsets": {k: dict(v) for k, v in offsets.items()},
                   "qa_by_domain": dict(sorted(qa_by_domain.items())),
                   "usable_by_domain": dict(sorted(usable_by_domain.items())),
                   "context_verified_by_domain": dict(sorted(context_verified_by_domain.items())),
                   "reason_counts": dict(reason_counts), "reference_counts": dict(reference_counts),
                   "context_counts": dict(context_counts), "qa_status": dict(qa_status),
                   "missing_examples": missing_examples,
                   "unresolved_context_examples": ambiguous_examples,
                   "covered_documents": len({q["doc_name"] for q in qas if q["doc_name"] in documents})})
    return result


def main():
    print(json.dumps(audit_qa(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
