import os
import json
import re

INPUT_FOLDER = "data/interim/extracted"
OUTPUT_FOLDER = "data/interim/cleaned"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text):

    # Only remove obvious extraction whitespace.
    # Do NOT modify numbers, units, chemical names, or terminology.

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith(".json"):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)

    print(f"Cleaning: {filename}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_pages = []

    for page in data["pages"]:

        original_text = page["text"]

        cleaned_text = clean_text(original_text)

        # Keep pages containing text
        if len(cleaned_text.strip()) > 0:

            cleaned_pages.append({
                "page": page["page"],
                "text": cleaned_text
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


print("\nConservative cleaning completed!")
