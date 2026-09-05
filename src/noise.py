import json
import random
from pathlib import Path


DEFAULT_SEED = 42


def create_clean_version(document: dict) -> dict:
    """
    Create a clean copy of the original document while preserving
    document and paragraph metadata.
    """
    return {
        "document_id": document["document_id"],
        "title": document["title"],
        "noise_level": "clean",
        "noise_seed": None,
        "paragraphs": [
            {
                "paragraph_id": paragraph["paragraph_id"],
                "text": paragraph["text"],
            }
            for paragraph in document["paragraphs"]
        ],
    }


def create_moderate_noise(
    document: dict,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Create a deterministic moderate formatting-noise version.

    The transformation changes formatting while attempting to
    preserve the original semantic information.

    Current pilot transformations:
    1. irregular whitespace
    2. unexpected line breaks

    Paragraph IDs remain unchanged so ground truth can still
    be traced to the original source paragraph.
    """
    rng = random.Random(seed)

    noisy_paragraphs = []

    for paragraph in document["paragraphs"]:
        original_text = paragraph["text"]

        words = original_text.split()

        formatted_words = []

        for word in words:
            formatted_words.append(word)

            # Occasionally create an additional whitespace gap.
            if rng.random() < 0.15:
                formatted_words.append("")

        noisy_text = " ".join(formatted_words)

        # Convert some spaces into unexpected line breaks.
        characters = list(noisy_text)

        space_positions = [
            index
            for index, character in enumerate(characters)
            if character == " "
        ]

        if space_positions:
            number_of_breaks = max(
                1,
                int(len(space_positions) * 0.08),
            )

            selected_positions = rng.sample(
                space_positions,
                min(number_of_breaks, len(space_positions)),
            )

            for position in selected_positions:
                characters[position] = "\n"

        noisy_text = "".join(characters)

        noisy_paragraphs.append(
            {
                "paragraph_id": paragraph["paragraph_id"],
                "text": noisy_text,
            }
        )

    return {
        "document_id": document["document_id"],
        "title": document["title"],
        "noise_level": "moderate",
        "noise_seed": seed,
        "paragraphs": noisy_paragraphs,
    }


def load_json(file_path: str) -> dict:
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict, file_path: str) -> None:
    path = Path(file_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

def normalize_whitespace(text: str) -> str:
    """
    Collapse spaces, tabs, and line breaks into single spaces.

    This allows us to check whether formatting corruption changed
    the underlying textual content.
    """
    return " ".join(text.split())


def validate_content_preservation(
    clean_document: dict,
    noisy_document: dict,
) -> bool:
    """
    Verify that formatting noise did not modify the underlying
    textual content of any paragraph.
    """
    if clean_document["document_id"] != noisy_document["document_id"]:
        return False

    if len(clean_document["paragraphs"]) != len(noisy_document["paragraphs"]):
        return False

    for clean_paragraph, noisy_paragraph in zip(
        clean_document["paragraphs"],
        noisy_document["paragraphs"],
    ):
        if (
            clean_paragraph["paragraph_id"]
            != noisy_paragraph["paragraph_id"]
        ):
            return False

        clean_text = normalize_whitespace(
            clean_paragraph["text"]
        )

        noisy_text = normalize_whitespace(
            noisy_paragraph["text"]
        )

        if clean_text != noisy_text:
            return False

    return True

if __name__ == "__main__":
    source_document = load_json(
        "data/raw/DOC_001.json"
    )

    clean_document = create_clean_version(
        source_document
    )

    moderate_document = create_moderate_noise(
        source_document,
        seed=42,
    )

    save_json(
        clean_document,
        "data/clean/DOC_001.json",
    )

    save_json(
        moderate_document,
        "data/moderate/DOC_001.json",
    )

    print("Created clean and moderate versions.")
    print()

    print("=== CLEAN P003 ===")
    print(clean_document["paragraphs"][2]["text"])

    print()

    print("=== MODERATE P003 ===")
    print(moderate_document["paragraphs"][2]["text"])

    print()

    print(
        "Clean paragraph ID:",
        clean_document["paragraphs"][2]["paragraph_id"],
    )

    print(
        "Moderate paragraph ID:",
        moderate_document["paragraphs"][2]["paragraph_id"],
    )

    print(
        "Noise seed:",
        moderate_document["noise_seed"],
    )
    content_preserved = validate_content_preservation(
        clean_document,
        moderate_document,
    )

    print(
        "Semantic text preserved:",
        content_preserved,
    )