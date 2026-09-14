"""Read-only structure-clean corpus audit; never embeds or builds an index."""

import json

from chunkers import structure_aware_chunk
from document_builder import build_document_text
from embeddings import load_embedding_model
from ohrbench_adapter import attach_page_provenance, iter_documents, load_pages


def main():
    pages = load_pages("clean")
    model = load_embedding_model()
    tokenizer = model.tokenizer
    limit = int(model.max_seq_length)
    special = tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = limit - special
    summary = {"documents": 0, "pages": len(pages), "chunks": 0,
               "model_max_seq_length": limit, "required_special_tokens": special,
               "empty_chunks": 0, "whitespace_only_chunks": 0,
               "provenance_validation_failures": 0, "source_span_failures": 0,
               "above_model_limit": 0, "maximum_content_tokens": 0,
               "maximum_actual_embedding_input_tokens": 0}
    for number, document in enumerate(iter_documents(pages, "clean"), 1):
        full_text, _ = build_document_text(document)
        chunks = structure_aware_chunk(document, tokenizer, content_limit)
        attach_page_provenance(chunks)
        for chunk in chunks:
            text = chunk["text"]
            summary["empty_chunks"] += not bool(text)
            summary["whitespace_only_chunks"] += not bool(text.strip())
            summary["provenance_validation_failures"] += not bool(
                chunk["source_page_ids"] and chunk["source_paragraph_ids"])
            start, end = chunk["start_char"], chunk["end_char"]
            summary["source_span_failures"] += not (
                0 <= start < end <= len(full_text) and full_text[start:end] == text)
            content = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            actual_input = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            summary["above_model_limit"] += actual_input > limit
            summary["maximum_content_tokens"] = max(summary["maximum_content_tokens"], content)
            summary["maximum_actual_embedding_input_tokens"] = max(
                summary["maximum_actual_embedding_input_tokens"], actual_input)
        summary["chunks"] += len(chunks)
        summary["documents"] = number
        if number % 50 == 0 or number == 1261:
            print(f"audited {number}/1261 documents; {summary['chunks']} chunks", flush=True)
    print(json.dumps(summary, indent=2))
    if any(summary[key] for key in ("empty_chunks", "whitespace_only_chunks",
                                    "provenance_validation_failures", "source_span_failures",
                                    "above_model_limit")):
        raise ValueError("Structure-clean corpus audit found invalid chunks")


if __name__ == "__main__":
    main()
