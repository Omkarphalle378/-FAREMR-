import os
import json
import re
import fitz


PDF_FOLDER = "data/raw/pdfs"
OUTPUT_FOLDER = "data/interim/structured"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# Clean a line without changing its actual content
# ---------------------------------------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Get all text lines with formatting information
# ---------------------------------------------------------

def extract_lines(page):

    blocks = page.get_text("dict").get("blocks", [])

    lines = []

    for block in blocks:

        if "lines" not in block:
            continue

        for line in block["lines"]:

            text = ""

            sizes = []
            flags = []

            y_position = line["bbox"][1]

            for span in line.get("spans", []):

                span_text = span.get("text", "")

                text += span_text

                if span_text.strip():

                    sizes.append(span.get("size", 0))
                    flags.append(span.get("flags", 0))

            text = clean_text(text)

            if not text or not sizes:
                continue

            lines.append({
                "text": text,
                "font_size": max(sizes),
                "flags": flags,
                "y": y_position
            })

    return lines


# ---------------------------------------------------------
# Determine whether a line looks like a heading
# ---------------------------------------------------------

def heading_score(line, normal_font_size):

    text = line["text"]
    size = line["font_size"]
    flags = line["flags"]

    score = 0

    words = text.split()
    word_count = len(words)

    # -----------------------------------------
    # 1. Font size
    # -----------------------------------------

    if normal_font_size > 0:

        ratio = size / normal_font_size

        if ratio >= 1.5:
            score += 4

        elif ratio >= 1.3:
            score += 3

        elif ratio >= 1.15:
            score += 1

    # -----------------------------------------
    # 2. Font flags
    #
    # PyMuPDF flag 16 commonly represents bold.
    # -----------------------------------------

    if any(flag & 16 for flag in flags):
        score += 2

    # -----------------------------------------
    # 3. Short line
    # -----------------------------------------

    if word_count <= 6:
        score += 2

    elif word_count <= 10:
        score += 1

    # -----------------------------------------
    # 4. Headings normally don't end
    # with sentence punctuation
    # -----------------------------------------

    if not text.endswith((".", ",", ";", ":", "?")):
        score += 1

    # -----------------------------------------
    # 5. Very long text is probably
    # a paragraph, not a heading
    # -----------------------------------------

    if word_count > 15:
        score -= 4

    if len(text) > 120:
        score -= 4

    # -----------------------------------------
    # 6. Complete sentence
    # -----------------------------------------

    if text.endswith("."):
        score -= 2

    # -----------------------------------------
    # 7. Avoid obvious page/header/footer text
    # -----------------------------------------

    lower = text.lower()

    unwanted = [
        "page ",
        "www.",
        "http",
        "email:",
        "telephone:",
        "copyright"
    ]

    for word in unwanted:

        if word in lower:
            score -= 5

    return score


# ---------------------------------------------------------
# Process one PDF
# ---------------------------------------------------------

def process_pdf(pdf_path, filename):

    sections = []

    try:

        with fitz.open(pdf_path) as doc:

            for page_number, page in enumerate(doc, start=1):

                lines = extract_lines(page)

                if not lines:
                    continue

                # -----------------------------------------
                # Estimate normal body font size
                # -----------------------------------------

                font_counts = {}

                for line in lines:

                    size = round(line["font_size"], 1)

                    font_counts[size] = (
                        font_counts.get(size, 0) + 1
                    )

                normal_font_size = max(
                    font_counts,
                    key=font_counts.get
                )

                # -----------------------------------------
                # Detect sections
                # -----------------------------------------

                current_section = "General"
                current_text = []

                for line in lines:

                    score = heading_score(
                        line,
                        normal_font_size
                    )

                    is_heading = score >= 5

                    if is_heading:

                        # Save previous section
                        if current_text:

                            text = " ".join(
                                current_text
                            ).strip()

                            if text:

                                sections.append({
                                    "source": filename,
                                    "page": page_number,
                                    "section": current_section,
                                    "text": text
                                })

                        # Start new section
                        current_section = line["text"]
                        current_text = []

                        print(
                            f"[HEADING] "
                            f"{filename} | "
                            f"Page {page_number} | "
                            f"{line['text']} | "
                            f"score={score}"
                        )

                    else:

                        current_text.append(
                            line["text"]
                        )

                # -----------------------------------------
                # Save last section on page
                # -----------------------------------------

                if current_text:

                    text = " ".join(
                        current_text
                    ).strip()

                    if text:

                        sections.append({
                            "source": filename,
                            "page": page_number,
                            "section": current_section,
                            "text": text
                        })

    except Exception as e:

        print(
            f"✗ ERROR processing {filename}: {e}"
        )

    return sections


# ---------------------------------------------------------
# Process ALL PDFs
# ---------------------------------------------------------

for filename in os.listdir(PDF_FOLDER):

    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(
        PDF_FOLDER,
        filename
    )

    print("\n" + "=" * 70)
    print("Processing:", filename)
    print("=" * 70)

    sections = process_pdf(
        pdf_path,
        filename
    )

    output_filename = (
        os.path.splitext(filename)[0] + ".json"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sections,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"✓ Saved {len(sections)} sections → "
        f"{output_filename}"
    )


print("\nAll PDFs processed!")
