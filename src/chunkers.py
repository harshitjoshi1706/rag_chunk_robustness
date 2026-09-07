import numpy as np

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

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
DEFAULT_SEMANTIC_THRESHOLD = 0.50
DEFAULT_SEMANTIC_MAX_TOKENS = 256


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

def get_sentence_spans(
    full_text: str,
) -> List[Dict]:
    """
    Split text into sentences while recovering each sentence's
    character span in the original document.
    """

    sentences = sent_tokenize(
        full_text
    )

    sentence_spans = []

    search_start = 0

    for sentence in sentences:
        start_char = full_text.find(
            sentence,
            search_start,
        )

        if start_char == -1:
            raise ValueError(
                "Could not recover sentence position "
                "inside the source document."
            )

        end_char = (
            start_char
            + len(sentence)
        )

        sentence_spans.append(
            {
                "text": sentence,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

        search_start = end_char

    return sentence_spans

def cosine_similarity_pair(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two vectors.
    """

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(
            vector_a,
            vector_b,
        )
        / denominator
    )

def semantic_chunk(
    document: Dict,
    tokenizer,
    embedding_model,
    similarity_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    max_tokens: int = DEFAULT_SEMANTIC_MAX_TOKENS,
) -> List[Dict]:
    """
    Create semantic chunks by comparing embeddings of consecutive
    sentences.

    A new chunk begins when:

    1. cosine similarity between consecutive sentences falls below
       the fixed threshold, OR
    2. adding the next sentence would exceed the maximum token budget.

    The same similarity threshold must be used across all formatting
    noise levels in the final experiment.
    """

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            "similarity_threshold must be between 0 and 1."
        )

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than zero."
        )

    full_text, paragraph_spans = (
        build_document_text(
            document
        )
    )

    sentence_spans = (
        get_sentence_spans(
            full_text
        )
    )

    if not sentence_spans:
        return []

    sentence_texts = [
        sentence["text"]
        for sentence in sentence_spans
    ]

    embeddings = embedding_model.encode(
        sentence_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    chunks = []

    current_sentence_indices = [
        0
    ]

    chunk_number = 0

    for sentence_index in range(
        1,
        len(sentence_spans),
    ):
        previous_embedding = (
            embeddings[
                sentence_index - 1
            ]
        )

        current_embedding = (
            embeddings[
                sentence_index
            ]
        )

        similarity = (
            cosine_similarity_pair(
                previous_embedding,
                current_embedding,
            )
        )

        candidate_indices = (
            current_sentence_indices
            + [sentence_index]
        )

        candidate_start = (
            sentence_spans[
                candidate_indices[0]
            ]["start_char"]
        )

        candidate_end = (
            sentence_spans[
                candidate_indices[-1]
            ]["end_char"]
        )

        candidate_text = (
            full_text[
                candidate_start:
                candidate_end
            ]
        )

        candidate_token_count = len(
            tokenizer(
                candidate_text,
                add_special_tokens=False,
            )["input_ids"]
        )

        semantic_break = (
            similarity
            < similarity_threshold
        )

        token_budget_break = (
            candidate_token_count
            > max_tokens
        )

        if (
            semantic_break
            or token_budget_break
        ):
            start_char = (
                sentence_spans[
                    current_sentence_indices[0]
                ]["start_char"]
            )

            end_char = (
                sentence_spans[
                    current_sentence_indices[-1]
                ]["end_char"]
            )

            chunk_text = (
                full_text[
                    start_char:end_char
                ]
            )

            token_count = len(
                tokenizer(
                    chunk_text,
                    add_special_tokens=False,
                )["input_ids"]
            )

            source_paragraph_ids = (
                get_paragraph_ids_for_span(
                    start_char,
                    end_char,
                    paragraph_spans,
                )
            )

            chunks.append(
                {
                    "chunk_id": (
                        f"{document['document_id']}"
                        f"_semantic_{chunk_number:04d}"
                    ),
                    "document_id": (
                        document["document_id"]
                    ),
                    "chunker": "semantic",
                    "text": chunk_text,
                    "token_count": token_count,
                    "start_char": start_char,
                    "end_char": end_char,
                    "source_paragraph_ids": (
                        source_paragraph_ids
                    ),
                }
            )

            chunk_number += 1

            current_sentence_indices = [
                sentence_index
            ]

        else:
            current_sentence_indices.append(
                sentence_index
            )

    # Save the final unfinished group.
    if current_sentence_indices:
        start_char = (
            sentence_spans[
                current_sentence_indices[0]
            ]["start_char"]
        )

        end_char = (
            sentence_spans[
                current_sentence_indices[-1]
            ]["end_char"]
        )

        chunk_text = (
            full_text[
                start_char:end_char
            ]
        )

        token_count = len(
            tokenizer(
                chunk_text,
                add_special_tokens=False,
            )["input_ids"]
        )

        source_paragraph_ids = (
            get_paragraph_ids_for_span(
                start_char,
                end_char,
                paragraph_spans,
            )
        )

        chunks.append(
            {
                "chunk_id": (
                    f"{document['document_id']}"
                    f"_semantic_{chunk_number:04d}"
                ),
                "document_id": (
                    document["document_id"]
                ),
                "chunker": "semantic",
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

def validate_semantic_chunks(
    chunks: List[Dict],
    max_tokens: int,
) -> bool:
    """
    Verify basic semantic-chunk correctness.
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

        if chunk["token_count"] > max_tokens:
            return False

        if chunk["end_char"] <= chunk["start_char"]:
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

    # ==================================================
    # SEMANTIC CHUNKING TEST
    # ==================================================

    print(
        "\n\n=== SEMANTIC CHUNKING TEST ==="
    )

    embedding_model = SentenceTransformer(
        MODEL_NAME
    )

    test_semantic_threshold = 0.50
    test_semantic_max_tokens = 40

    semantic_chunks = semantic_chunk(
        document=document,
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        similarity_threshold=(
            test_semantic_threshold
        ),
        max_tokens=(
            test_semantic_max_tokens
        ),
    )

    print(
        "Document:",
        document["document_id"],
    )

    print(
        "Debug similarity threshold:",
        test_semantic_threshold,
    )

    print(
        "Debug max tokens:",
        test_semantic_max_tokens,
    )

    print(
        "Number of semantic chunks:",
        len(semantic_chunks),
    )

    for chunk in semantic_chunks:
        print(
            "\n------------------------------"
        )

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

    semantic_validation_passed = (
        validate_semantic_chunks(
            chunks=semantic_chunks,
            max_tokens=(
                test_semantic_max_tokens
            ),
        )
    )

    print(
        "\nSemantic validation passed:",
        semantic_validation_passed,
    )