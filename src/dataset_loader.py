import json
from pathlib import Path


def load_json(file_path: str):
    """
    Load JSON data from disk.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_document(file_path: str) -> dict:
    """
    Load a single document JSON file.
    """
    document = load_json(file_path)

    required_keys = {"document_id", "title", "paragraphs"}

    if not required_keys.issubset(document.keys()):
        raise ValueError(
            f"Document is missing required keys: {required_keys}"
        )

    return document


def load_questions(file_path: str) -> list:
    """
    Load question and ground-truth annotations.
    """
    questions = load_json(file_path)

    if not isinstance(questions, list):
        raise ValueError("Questions file must contain a JSON list.")

    return questions


if __name__ == "__main__":
    document = load_document("data/raw/DOC_001.json")
    questions = load_questions("data/raw/questions.json")

    print("=== DOCUMENT ===")
    print("Document ID:", document["document_id"])
    print("Title:", document["title"])
    print("Paragraph count:", len(document["paragraphs"]))

    print("\n=== QUESTIONS ===")
    print("Question count:", len(questions))

    for question in questions:
        print("Question ID:", question["question_id"])
        print("Question:", question["question"])
        print(
            "Relevant document:",
            question["relevant_document_id"]
        )
        print(
            "Relevant paragraphs:",
            question["relevant_paragraph_ids"]
        )