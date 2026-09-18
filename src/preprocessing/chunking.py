"""Semantic sentence-aware chunking with sliding sentence overlap."""

import json
from pathlib import Path
import re

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "structured"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "chunks"


def normalize_text(text: str) -> str:
    """Clean and normalize whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list:
    """Split text into sentences while retaining sentence integrity."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def create_chunks(text: str, max_words: int = 250, overlap_sentences: int = 1) -> list:
    """Create chunks without breaking sentences, maintaining a sliding sentence overlap."""
    sentences = split_sentences(text)
    chunks = []

    current = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # If adding this sentence exceeds the limit, save current chunk first
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))

            # Keep the last sentence(s) as overlap
            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                current_words = len(" ".join(current).split())
            else:
                current = []
                current_words = 0

        current.append(sentence)
        current_words += sentence_words

    # Save remaining text
    if current:
        chunks.append(" ".join(current))

    return chunks


def run_chunking(input_dir: Path = INPUT_FOLDER, output_dir: Path = OUTPUT_FOLDER) -> None:
    """Process all structured JSON files and generate indexed chunk files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return

    json_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".json"])
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    global_chunk_id = 1

    for json_file in json_files:
        filename = json_file.name
        print(f"\nProcessing: {filename}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        output_chunks = []

        for record in data:
            source = record.get("source", "")
            page = record.get("page")
            section = record.get("section", "General")
            text = normalize_text(record.get("text", ""))

            if not text:
                continue

            chunks = create_chunks(text, max_words=250, overlap_sentences=1)

            for chunk in chunks:
                output_chunks.append({
                    "id": f"chunk_{global_chunk_id:06d}",
                    "source": source,
                    "page": page,
                    "section": section,
                    "text": chunk
                })
                global_chunk_id += 1

        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_chunks, f, ensure_ascii=False, indent=2)

        print(f"Created {len(output_chunks)} chunks")

    print("\nChunking completed!")


def main():
    run_chunking()


if __name__ == "__main__":
    main()
