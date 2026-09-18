"""Standard text cleaning pipeline for agricultural document pages."""

import json
from pathlib import Path
import re

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "conservative"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "cleaned"


def clean_text(text: str) -> str:
    """Normalize whitespace, punctuation spacing, and line breaks."""
    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Fix spaces around slash
    text = re.sub(r"\s*/\s*", "/", text)
    # Remove excessive newlines
    text = re.sub(r"\n+", "\n", text)
    # Remove spaces at beginning/end
    return text.strip()


def run_clean_text(input_dir: Path = INPUT_FOLDER, output_dir: Path = OUTPUT_FOLDER) -> None:
    """Process all conservative JSON files and output cleaned text pages."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return

    json_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".json"])
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    for json_file in json_files:
        filename = json_file.name
        print(f"Cleaning: {filename}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_pages = []
        for page in data.get("pages", []):
            text = clean_text(page.get("text", ""))

            # Skip pages with no meaningful text
            if len(text) < 30:
                continue

            cleaned_pages.append({
                "page": page.get("page"),
                "text": text
            })

        output = {
            "source": data.get("source", filename),
            "pages": cleaned_pages
        }

        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved: {filename} ({len(cleaned_pages)} pages)")

    print("\nCleaning completed!")


def main():
    run_clean_text()


if __name__ == "__main__":
    main()