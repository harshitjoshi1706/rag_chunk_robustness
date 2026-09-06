from typing import List, Dict, Tuple


def build_document_text(document: dict) -> Tuple[str, List[Dict]]:
    """
    Combine all paragraphs into one document string while recording
    the character span occupied by each original paragraph.

    Returns:
        full_text:
            Combined document text.

        paragraph_spans:
            Metadata describing where each source paragraph occurs
            inside the combined text.
    """

    text_parts = []
    paragraph_spans = []

    current_position = 0

    for index, paragraph in enumerate(document["paragraphs"]):
        paragraph_id = paragraph["paragraph_id"]
        paragraph_text = paragraph["text"]

        start_char = current_position

        text_parts.append(paragraph_text)

        current_position += len(paragraph_text)

        end_char = current_position

        paragraph_spans.append(
            {
                "document_id": document["document_id"],
                "paragraph_id": paragraph_id,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

        # Separate paragraphs using two line breaks.
        if index < len(document["paragraphs"]) - 1:
            separator = "\n\n"
            text_parts.append(separator)
            current_position += len(separator)

    full_text = "".join(text_parts)

    return full_text, paragraph_spans


def get_paragraph_ids_for_span(
    chunk_start: int,
    chunk_end: int,
    paragraph_spans: List[Dict],
) -> List[str]:
    """
    Determine which original paragraphs overlap a given character span.
    """

    paragraph_ids = []

    for span in paragraph_spans:
        paragraph_start = span["start_char"]
        paragraph_end = span["end_char"]

        overlaps = (
            chunk_start < paragraph_end
            and chunk_end > paragraph_start
        )

        if overlaps:
            paragraph_ids.append(span["paragraph_id"])

    return paragraph_ids

if __name__ == "__main__":
    from dataset_loader import load_document

    document = load_document(
        "data/clean/DOC_001.json"
    )

    full_text, paragraph_spans = build_document_text(
        document
    )

    print("=== FULL DOCUMENT TEXT ===")
    print(full_text)

    print("\n=== PARAGRAPH SPANS ===")

    for span in paragraph_spans:
        print(span)

    # Test using paragraph P003.
    p003 = paragraph_spans[2]

    detected_ids = get_paragraph_ids_for_span(
        p003["start_char"],
        p003["end_char"],
        paragraph_spans,
    )

    print("\nTest span belongs to:", detected_ids)