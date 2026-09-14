"""Read-only clean-corpus preflight for recursive chunks before embedding."""

import json

from chunkers import DEFAULT_FIXED_CHUNK_SIZE, load_tokenizer, recursive_chunk
from document_builder import build_document_text
from ohrbench_adapter import attach_page_provenance, iter_documents, load_pages


def main():
    pages = load_pages("clean")
    tokenizer = load_tokenizer()
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if special_tokens != 2:
        raise RuntimeError(f"Unexpected special-token count: {special_tokens}")
    summary = {"documents": 0, "pages": len(pages), "chunks": 0,
               "empty_chunks": 0, "whitespace_only_chunks": 0,
               "missing_provenance": 0, "maximum_content_tokens": 0,
               "maximum_embedding_input_tokens": 0}
    for number, document in enumerate(iter_documents(pages, "clean"), 1):
        full_text, _ = build_document_text(document)
        chunks = recursive_chunk(document, tokenizer)
        attach_page_provenance(chunks)
        previous_start = -1
        for chunk in chunks:
            text = chunk["text"]
            if not text:
                summary["empty_chunks"] += 1
            if not text.strip():
                summary["whitespace_only_chunks"] += 1
            if not chunk["source_page_ids"] or not chunk["source_paragraph_ids"]:
                summary["missing_provenance"] += 1
            start, end = chunk["start_char"], chunk["end_char"]
            if not (0 <= start < end <= len(full_text)) or full_text[start:end] != text:
                raise ValueError(f"Source span mismatch: {chunk['chunk_id']}")
            if start <= previous_start:
                raise ValueError(f"Nonmonotonic source position: {chunk['chunk_id']}")
            previous_start = start
            actual = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if actual != chunk["token_count"] or actual > DEFAULT_FIXED_CHUNK_SIZE:
                raise ValueError(f"Invalid token count: {chunk['chunk_id']} ({actual})")
            summary["maximum_content_tokens"] = max(summary["maximum_content_tokens"], actual)
            summary["maximum_embedding_input_tokens"] = max(
                summary["maximum_embedding_input_tokens"], actual + special_tokens)
        summary["chunks"] += len(chunks)
        summary["documents"] = number
        if number % 50 == 0 or number == 1261:
            print(f"audited {number}/1261 documents; {summary['chunks']} chunks", flush=True)
    if any(summary[key] for key in ("empty_chunks", "whitespace_only_chunks", "missing_provenance")):
        raise ValueError(f"Recursive corpus preflight failed: {summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
