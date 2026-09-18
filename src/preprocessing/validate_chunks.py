"""Validation and structural integrity checks for text chunks."""

from collections import Counter
import json
from pathlib import Path
import re

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "chunks"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "validated"


def validate_chunk(chunk: dict) -> list:
    """Check a chunk for missing metadata, extreme length, or extraction artifacts."""
    problems = []

    source = chunk.get("source")
    page = chunk.get("page")
    section = chunk.get("section")
    text = chunk.get("text")

    # Missing metadata
    if not source:
        problems.append("missing_source")
    if page is None:
        problems.append("missing_page")
    if not section:
        problems.append("missing_section")

    # Empty text
    if not text:
        problems.append("empty_text")
        return problems

    # Length checks
    word_count = len(text.split())
    if word_count < 10:
        problems.append("very_short")
    if word_count > 400:
        problems.append("very_long")

    # Suspicious extraction artifacts
    if "" in text:
        problems.append("invalid_character")
    if re.search(r"(.)\1{5,}", text):
        problems.append("repeated_characters")

    # Alphanumeric ratio check
    if len(text) > 0:
        alphanumeric = sum(c.isalnum() or c.isspace() for c in text)
        normal_ratio = alphanumeric / len(text)
        if normal_ratio < 0.65:
            problems.append("many_special_characters")

    return problems


def run_validation(input_dir: Path = INPUT_FOLDER, output_dir: Path = OUTPUT_FOLDER) -> None:
    """Validate all chunk files, isolate flagged issues, and report duplicate chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return

    json_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".json"])
    if not json_files:
        print(f"No chunk JSON files found in {input_dir}")
        return

    all_chunks = []
    all_problems = []

    for json_file in json_files:
        filename = json_file.name
        print(f"\nChecking: {filename}")

        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        valid_chunks = []
        for chunk in chunks:
            problems = validate_chunk(chunk)
            if problems:
                all_problems.append({
                    "id": chunk.get("id"),
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "problems": problems,
                    "text": chunk.get("text", "")
                })
            else:
                valid_chunks.append(chunk)
                all_chunks.append(chunk)

        # Save validated chunks for this PDF
        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(valid_chunks, f, ensure_ascii=False, indent=2)

        print(
            f"Total: {len(chunks)} | "
            f"Passed: {len(valid_chunks)} | "
            f"Flagged: {len(chunks) - len(valid_chunks)}"
        )

    # Check duplicate chunks across all files
    texts = [chunk["text"].strip().lower() for chunk in all_chunks]
    counts = Counter(texts)

    duplicates = []
    for chunk in all_chunks:
        text_key = chunk["text"].strip().lower()
        if counts[text_key] > 1:
            duplicates.append({
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "section": chunk["section"],
                "text": chunk["text"],
                "duplicate_count": counts[text_key]
            })

    # Save problem report
    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(all_problems, f, ensure_ascii=False, indent=2)

    # Save duplicate report
    with open(output_dir / "duplicates.json", "w", encoding="utf-8") as f:
        json.dump(duplicates, f, ensure_ascii=False, indent=2)

    # Save combined validated chunks before deduplication
    with open(output_dir / "all_chunks_before_dedup.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Total valid chunks: {len(all_chunks)}")
    print(f"Flagged chunks: {len(all_problems)}")
    print(f"Duplicate chunks: {len(duplicates)}")
    print("\nReports created in data/interim/validated/:")
    print("├── validation_report.json")
    print("├── duplicates.json")
    print("└── all_chunks_before_dedup.json")


def main():
    run_validation()


if __name__ == "__main__":
    main()
