import os
import json
import fitz


# Input and output folders
PDF_FOLDER = "data/raw/pdfs"
OUTPUT_FOLDER = "data/interim/extracted"


# Create output folder if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Get all files from the PDF folder
pdf_files = os.listdir(PDF_FOLDER)


for filename in pdf_files:

    # Process only PDF files
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_FOLDER, filename)

    try:
        # Open PDF
        doc = fitz.open(pdf_path)

        pages = []

        # Process each page
        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")
            text = text.strip()

            # Skip pages without meaningful text
            if not text:
                continue

            pages.append({
                "page": page_number,
                "text": text
            })

        # Create output structure
        output = {
            "source": filename,
            "pages": pages
        }

        # Create JSON filename
        output_filename = os.path.splitext(filename)[0] + ".json"

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_filename
        )

        # Save extracted data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=2
            )

        doc.close()

        print(f"Saved: {output_filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")


print("\nAll PDFs extracted!")
