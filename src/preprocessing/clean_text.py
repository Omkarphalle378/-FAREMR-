import os
import json
import re


INPUT_FOLDER = "data/interim/conservative"
OUTPUT_FOLDER = "data/interim/cleaned"


# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text):

    # Remove excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)

    # Fix spaces around slash
    text = re.sub(r'\s*/\s*', '/', text)

    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)

    # Remove spaces at beginning/end
    text = text.strip()

    return text


for filename in os.listdir(INPUT_FOLDER):

    # Process only JSON files
    if not filename.lower().endswith(".json"):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)

    print(f"Cleaning: {filename}")

    # Read input JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_pages = []

    for page in data["pages"]:

        text = clean_text(page["text"])

        # Skip pages with no meaningful text
        if len(text) < 30:
            continue

        cleaned_pages.append({
            "page": page["page"],
            "text": text
        })

    output = {
        "source": data["source"],
        "pages": cleaned_pages
    }

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"✓ Saved: {filename}")


print("\nCleaning completed!")