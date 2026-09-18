"""Final chunk cleaning, artifact filtering, and duplicate removal."""

import json
from pathlib import Path
import re

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "validated" / "all_chunks_before_dedup.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "validated" / "final_chunks.json"
REMOVED_FILE = PROJECT_ROOT / "data" / "interim" / "validated" / "removed_chunks.json"


def normalize_text(text: str) -> str:
    """Normalize text only for duplicate comparison. Original chunk text is preserved."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_junk(chunk: dict):
    """Remove obvious extraction junk without discarding agricultural knowledge."""
    text = chunk.get("text", "").strip()

    if not text:
        return True, "empty"

    # Completely corrupted replacement character
    if "" in text:
        return True, "invalid_character"

    # Only page numbers
    if re.fullmatch(r"\d+", text):
        return True, "page_number"

    # Obvious separator lines
    if re.fullmatch(r"[_=\-]{3,}", text):
        return True, "separator"

    # Very short symbol-only text
    if len(text) <= 3 and not re.search(r"[A-Za-z\u0900-\u097F]", text):
        return True, "symbol_only"

    # Obvious repeated website/footer information
    lower = text.lower()
    junk_patterns = [
        "copyright",
        "www.",
        "http://",
        "https://",
    ]

    for pattern in junk_patterns:
        if pattern in lower:
            return True, "footer_or_metadata"

    return False, None


def run_finalization(
    input_path: Path = INPUT_FILE,
    output_path: Path = OUTPUT_FILE,
    removed_path: Path = REMOVED_FILE
) -> None:
    """Filter junk and duplicate chunks to produce the final knowledge base chunks."""
    print("=" * 60)
    print("FINAL CHUNK CLEANING")
    print("=" * 60)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    # Load combined chunks
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Input chunks : {len(chunks)}")

    final_chunks = []
    removed_chunks = []
    seen = set()

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        # 1. Remove obvious junk
        junk, reason = is_junk(chunk)
        if junk:
            removed_chunks.append({
                "chunk": chunk,
                "reason": reason
            })
            continue

        # 2. Exact/normalized duplicate detection
        normalized = normalize_text(text)
        if normalized in seen:
            removed_chunks.append({
                "chunk": chunk,
                "reason": "duplicate"
            })
            continue

        seen.add(normalized)

        # 3. Keep the original chunk unchanged
        final_chunks.append(chunk)

    # Save final chunks
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)

    # Save removed chunks for audit
    with open(removed_path, "w", encoding="utf-8") as f:
        json.dump(removed_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("FINAL CHUNK CLEANING COMPLETE")
    print("=" * 60)
    print(f"Input chunks       : {len(chunks)}")
    print(f"Final chunks       : {len(final_chunks)}")
    print(f"Removed chunks     : {len(removed_chunks)}")
    print(f"\nSaved:\n{output_path}\n{removed_path}")
    print("\nReady for BGE-M3 embeddings.")


def main():
    run_finalization()


if __name__ == "__main__":
    main()
