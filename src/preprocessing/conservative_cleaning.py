"""Conservative text cleaning for extracted PDF pages.

Removes extraction whitespace artifacts while strictly preserving numbers, units,
chemical formulations, and Marathi/English agricultural terminology.
"""

import json
from pathlib import Path
import re

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "extracted"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "conservative"


def clean_text(text: str) -> str:
    """Only remove obvious extraction whitespace without altering terminology."""
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading/trailing whitespace
    return text.strip()


def run_conservative_cleaning(input_dir: Path = INPUT_FOLDER, output_dir: Path = OUTPUT_FOLDER) -> None:
    """Process all extracted JSON files with conservative cleaning rules."""
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
            original_text = page.get("text", "")
            cleaned_text = clean_text(original_text)

            # Keep pages containing text
            if len(cleaned_text.strip()) > 0:
                cleaned_pages.append({
                    "page": page.get("page"),
                    "text": cleaned_text
                })

        output = {
            "source": data.get("source", filename),
            "pages": cleaned_pages
        }

        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved: {filename} ({len(cleaned_pages)} pages)")

    print("\nConservative cleaning completed!")


def main():
    run_conservative_cleaning()


if __name__ == "__main__":
    main()
