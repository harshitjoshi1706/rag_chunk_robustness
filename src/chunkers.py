from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, List

from transformers import AutoTokenizer

from dataset_loader import load_document
from document_builder import (
    build_document_text,
    get_paragraph_ids_for_span,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_FIXED_CHUNK_SIZE = 256
DEFAULT_FIXED_CHUNK_OVERLAP = 32


def load_tokenizer():
    """
    Load the tokenizer associated with the retrieval embedding model.

    A fast tokenizer is required because we need token-to-character
    offsets for provenance tracking.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset mapping."
        )

    return tokenizer


def fixed_size_chunk(
    document: Dict,
    tokenizer,
    chunk_size: int = DEFAULT_FIXED_CHUNK_SIZE,
    overlap: int = DEFAULT_FIXED_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split a document into fixed-size token chunks while preserving
    source paragraph provenance.

    Args:
        document:
            Document dictionary containing document_id and paragraphs.

        tokenizer:
            Hugging Face fast tokenizer.

        chunk_size:
            Maximum number of tokens in each chunk.

        overlap:
            Number of tokens repeated between consecutive chunks.

    Returns:
        List of chunk dictionaries.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    full_text, paragraph_spans = (
        build_document_text(document)
    )

    encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )

    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    chunks = []

    step_size = chunk_size - overlap

    chunk_number = 0

    for token_start in range(
        0,
        len(token_ids),
        step_size,
    ):
        token_end = min(
            token_start + chunk_size,
            len(token_ids),
        )

        chunk_offsets = offsets[
            token_start:token_end
        ]

        if not chunk_offsets:
            break

        start_char = chunk_offsets[0][0]
        end_char = chunk_offsets[-1][1]

        chunk_text = full_text[
            start_char:end_char
        ]

        source_paragraph_ids = (
            get_paragraph_ids_for_span(
                start_char,
                end_char,
                paragraph_spans,
            )
        )

        chunk = {
            "chunk_id": (
                f"{document['document_id']}"
                f"_fixed_{chunk_number:04d}"
            ),
            "document_id": document[
                "document_id"
            ],
            "chunker": "fixed",
            "text": chunk_text,
            "token_start": token_start,
            "token_end": token_end,
            "token_count": (
                token_end - token_start
            ),
            "start_char": start_char,
            "end_char": end_char,
            "source_paragraph_ids": (
                source_paragraph_ids
            ),
        }

        chunks.append(chunk)

        chunk_number += 1

        # Stop once the final token has been included.
        if token_end == len(token_ids):
            break

    return chunks

def recursive_chunk(
    document: Dict,
    tokenizer,
    chunk_size: int = DEFAULT_FIXED_CHUNK_SIZE,
    overlap: int = DEFAULT_FIXED_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Split a document recursively using natural text boundaries while
    enforcing a token-based chunk-size budget.

    Boundary preference:
    paragraph -> line break -> sentence-like boundary -> space
    -> character fallback.

    Source paragraph provenance is preserved using character offsets.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    full_text, paragraph_spans = build_document_text(
        document
    )

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        add_start_index=True,
        strip_whitespace=False,
    )

    split_documents = splitter.create_documents(
        [full_text]
    )

    chunks = []

    for chunk_number, split_document in enumerate(
        split_documents
    ):
        chunk_text = split_document.page_content

        start_char = split_document.metadata[
            "start_index"
        ]

        end_char = (
            start_char
            + len(chunk_text)
        )

        source_paragraph_ids = (
            get_paragraph_ids_for_span(
                start_char,
                end_char,
                paragraph_spans,
            )
        )

        token_count = len(
            tokenizer(
                chunk_text,
                add_special_tokens=False,
            )["input_ids"]
        )

        chunks.append(
            {
                "chunk_id": (
                    f"{document['document_id']}"
                    f"_recursive_{chunk_number:04d}"
                ),
                "document_id": document[
                    "document_id"
                ],
                "chunker": "recursive",
                "text": chunk_text,
                "token_count": token_count,
                "start_char": start_char,
                "end_char": end_char,
                "source_paragraph_ids": (
                    source_paragraph_ids
                ),
            }
        )

    return chunks

def validate_recursive_chunks(
    chunks: List[Dict],
    chunk_size: int,
) -> bool:
    """
    Verify basic correctness of recursively generated chunks.
    """

    if not chunks:
        return False

    for chunk in chunks:
        if not chunk["text"].strip():
            return False

        if not chunk["source_paragraph_ids"]:
            return False

        if chunk["token_count"] <= 0:
            return False

        if chunk["token_count"] > chunk_size:
            return False

        if chunk["end_char"] <= chunk["start_char"]:
            return False

    return True

def validate_fixed_chunks(
    chunks: List[Dict],
    chunk_size: int,
    overlap: int,
) -> bool:
    """
    Validate the fixed-size chunk sequence.

    Checks:
    - chunks are not empty
    - provenance exists
    - non-final chunks have the expected token count
    - consecutive chunks use the expected overlap
    """

    if not chunks:
        return False

    expected_step = chunk_size - overlap

    for index, chunk in enumerate(chunks):
        if not chunk["text"].strip():
            return False

        if not chunk["source_paragraph_ids"]:
            return False

        if chunk["token_end"] <= chunk["token_start"]:
            return False

        # Every chunk except the final chunk should use
        # the complete target chunk size.
        if index < len(chunks) - 1:
            if chunk["token_count"] != chunk_size:
                return False

        # Check spacing between consecutive chunk starts.
        if index > 0:
            previous_chunk = chunks[index - 1]

            actual_step = (
                chunk["token_start"]
                - previous_chunk["token_start"]
            )

            if actual_step != expected_step:
                return False

    return True


if __name__ == "__main__":
    tokenizer = load_tokenizer()

    document = load_document(
        "data/clean/DOC_001.json"
    )

    # DEBUG configuration only.
    # Final experiment configuration:
    # 256-token chunks with 32-token overlap.
    test_chunk_size = 40
    test_overlap = 8

    # ==================================================
    # FIXED-SIZE CHUNKING TEST
    # ==================================================

    fixed_chunks = fixed_size_chunk(
        document=document,
        tokenizer=tokenizer,
        chunk_size=test_chunk_size,
        overlap=test_overlap,
    )

    print("=== FIXED-SIZE CHUNKING TEST ===")

    print(
        "Document:",
        document["document_id"],
    )

    print(
        "Debug chunk size:",
        test_chunk_size,
    )

    print(
        "Debug overlap:",
        test_overlap,
    )

    print(
        "Number of chunks:",
        len(fixed_chunks),
    )

    for chunk in fixed_chunks:
        print("\n------------------------------")

        print(
            "Chunk ID:",
            chunk["chunk_id"],
        )

        print(
            "Token range:",
            chunk["token_start"],
            "to",
            chunk["token_end"],
        )

        print(
            "Token count:",
            chunk["token_count"],
        )

        print(
            "Character range:",
            chunk["start_char"],
            "to",
            chunk["end_char"],
        )

        print(
            "Source paragraphs:",
            chunk["source_paragraph_ids"],
        )

        print(
            "Text:",
            repr(chunk["text"]),
        )

    fixed_validation_passed = validate_fixed_chunks(
        chunks=fixed_chunks,
        chunk_size=test_chunk_size,
        overlap=test_overlap,
    )

    print(
        "\nFixed-size validation passed:",
        fixed_validation_passed,
    )

    # ==================================================
    # RECURSIVE CHUNKING TEST
    # ==================================================

    print(
        "\n\n=== RECURSIVE CHUNKING TEST ==="
    )

    recursive_chunks = recursive_chunk(
        document=document,
        tokenizer=tokenizer,
        chunk_size=test_chunk_size,
        overlap=test_overlap,
    )

    print(
        "Document:",
        document["document_id"],
    )

    print(
        "Debug chunk size:",
        test_chunk_size,
    )

    print(
        "Debug overlap:",
        test_overlap,
    )

    print(
        "Number of recursive chunks:",
        len(recursive_chunks),
    )

    for chunk in recursive_chunks:
        print("\n------------------------------")

        print(
            "Chunk ID:",
            chunk["chunk_id"],
        )

        print(
            "Token count:",
            chunk["token_count"],
        )

        print(
            "Character range:",
            chunk["start_char"],
            "to",
            chunk["end_char"],
        )

        print(
            "Source paragraphs:",
            chunk["source_paragraph_ids"],
        )

        print(
            "Text:",
            repr(chunk["text"]),
        )

    recursive_validation_passed = (
        validate_recursive_chunks(
            chunks=recursive_chunks,
            chunk_size=test_chunk_size,
        )
    )

    print(
        "\nRecursive validation passed:",
        recursive_validation_passed,
    )