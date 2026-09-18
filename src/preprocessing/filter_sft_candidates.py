"""Agricultural knowledge indicator filtering for fine-tuning candidate chunks."""

import json
from pathlib import Path
import re

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "classified" / "sft_candidates.json"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "classified"
OUTPUT_FILE = OUTPUT_FOLDER / "filtered_sft_candidates.json"
REMOVED_FILE = OUTPUT_FOLDER / "removed_sft_candidates.json"


def normalize_text(text: str) -> str:
    """Normalize text whitespace and casing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Agricultural knowledge indicators
AGRICULTURAL_PATTERNS = [
    # Crop / planting
    r"\bsugarcane\b",
    r"\bplanting\b",
    r"\bplanted\b",
    r"\bplant\b",
    r"\bsowing\b",
    r"\bseed\b",
    r"\bseed cane\b",
    r"\bsett\b",
    r"\bsetts\b",
    r"\bsettling\b",
    r"\bgermination\b",

    # Varieties
    r"\bvariet",
    r"\bcultivar\b",
    r"\bhybrid\b",

    # Soil
    r"\bsoil\b",
    r"\bsoil type\b",
    r"\bsoil fertility\b",
    r"\bsoil moisture\b",
    r"\bdrainage\b",
    r"\bph\b",
    r"\bplough",
    r"\bplanking\b",
    r"\bharrowing\b",
    r"\bsubsoiling\b",
    r"\bsub-soiling\b",

    # Nutrients / fertilizer
    r"\bfertil",
    r"\bnitrogen\b",
    r"\bphosphorus\b",
    r"\bpotassium\b",
    r"\bnutrient\b",
    r"\bmanure\b",
    r"\bcompost\b",
    r"\bmicronutrient\b",
    r"\burea\b",

    # Water
    r"\birrigat",
    r"\bwater\b",
    r"\bwatering\b",
    r"\bmoisture\b",
    r"\bdrought\b",
    r"\bwaterlogging\b",
    r"\bwater logging\b",

    # Weed
    r"\bweed",
    r"\bherbicide\b",
    r"\bweeding\b",

    # Pest
    r"\bpest\b",
    r"\binsect\b",
    r"\binsect-pest\b",
    r"\bborer\b",
    r"\bwhitefly\b",
    r"\baphid\b",
    r"\btermite\b",

    # Disease
    r"\bdisease\b",
    r"\bfung",
    r"\bbacterial\b",
    r"\bvirus\b",
    r"\bviral\b",
    r"\bpathogen\b",
    r"\bred rot\b",
    r"\bsmut\b",
    r"\bwilt\b",

    # Crop stages
    r"\bgermination\b",
    r"\bemergence\b",
    r"\btillering\b",
    r"\bstem elongation\b",
    r"\bgrand growth\b",
    r"\bripening\b",
    r"\bmaturity\b",

    # Ratoon
    r"\bratoon\b",
    r"\bratooning\b",

    # Harvest
    r"\bharvest",
    r"\bcutting\b",
    r"\bmillable cane\b",

    # Climate
    r"\bclimat",
    r"\btemperature\b",
    r"\brainfall\b",
    r"\bhumidity\b",
    r"\bsunshine\b",
    r"\bfrost\b",
    r"\bflood\b",

    # Management / recommendations
    r"\bmanagement\b",
    r"\bcontrol\b",
    r"\brecommend",
    r"\bapplication\b",
    r"\bapply\b",
    r"\btreatment\b",
    r"\bschedule\b",
    r"\bmethod\b",
    r"\bpractice\b",

    # Measurements
    r"\bt\/ha\b",
    r"\bkg\/ha\b",
    r"\bkg per ha\b",
    r"\bkg\b",
    r"\btons?\b",
    r"\btonnes?\b",
    r"\bmm\b",
    r"\bcm\b",
    r"\bmg\b",
    r"\blitre\b",
    r"\bliter\b",
    r"\bpercent\b",
    r"\b°c\b",
    r"\bdegree",
]


def agricultural_score(chunk: dict):
    """Compute agricultural indicator density score for a candidate chunk."""
    text = normalize_text(chunk.get("text", ""))
    section = normalize_text(chunk.get("section", ""))
    combined = f"{section} {text}"

    score = 0
    matched_patterns = []

    for pattern in AGRICULTURAL_PATTERNS:
        if re.search(pattern, combined):
            score += 1
            matched_patterns.append(pattern)

    return score, matched_patterns


# Administrative sections
ADMINISTRATIVE_SECTIONS = [
    "foreword",
    "preface",
    "acknowledgement",
    "acknowledgments",
    "acknowledgements",
    "table of contents",
    "contents",
    "project description",
    "project objectives",
    "document information",
    "author information",
    "publication information",
]


def is_administrative_section(chunk: dict):
    section = normalize_text(chunk.get("section", ""))
    for pattern in ADMINISTRATIVE_SECTIONS:
        if pattern in section:
            return True, "administrative_section"
    return False, None


# Publication metadata
METADATA_PATTERNS = [
    r"\bisbn\b",
    r"\bissn\b",
    r"\bdoi\b",
    r"\bresearchgate\b",
    r"\bimpact factor\b",
    r"\bcopyright\b",
    r"\bpublished by\b",
    r"\bprinted by\b",
    r"\bpublication number\b",
    r"\btelephone\b",
    r"\bphone number\b",
    r"\bcontact information\b",
    r"\bwww\.",
    r"\bhttps?://",
]


def contains_metadata(chunk: dict):
    text = normalize_text(chunk.get("text", ""))
    section = normalize_text(chunk.get("section", ""))
    combined = f"{section} {text}"

    for pattern in METADATA_PATTERNS:
        if re.search(pattern, combined):
            return True, "publication_metadata"
    return False, None


def is_table_of_contents(chunk: dict):
    text = normalize_text(chunk.get("text", ""))
    section = normalize_text(chunk.get("section", ""))

    if not text:
        return False, None

    if section in ["table of contents", "contents", "index"]:
        return True, "table_of_contents"

    # Dotted TOC entry: Sugarcane varieties ........ 20
    dotted_entries = re.findall(r"\.{3,}\s*\d{1,3}", text)
    if len(dotted_entries) >= 1:
        return True, "table_of_contents"

    # Numbered TOC entries
    numbered_entries = re.findall(r"(?:^|\s)\d+\.\s+[A-Za-z][^0-9]{1,80}\s+\d{1,3}(?:\s|$)", text)
    if len(numbered_entries) >= 3:
        return True, "table_of_contents"

    return False, None


def is_heading_only(chunk: dict):
    text = chunk.get("text", "").strip()
    if not text:
        return True, "empty_text"

    words = text.split()
    if len(words) <= 5:
        has_number = bool(re.search(r"\d", text))
        has_sentence_punctuation = bool(re.search(r"[.!?]", text))
        score, _ = agricultural_score(chunk)

        if not has_number and not has_sentence_punctuation and score == 0:
            return True, "heading_only"

    return False, None


def is_low_information(chunk: dict):
    text = chunk.get("text", "").strip()
    if not text:
        return True, "empty_text"

    if len(text) < 20:
        score, _ = agricultural_score(chunk)
        if score < 2:
            return True, "too_short"

    return False, None


def is_reference_like(chunk: dict):
    text = normalize_text(chunk.get("text", ""))
    if not text:
        return False, None

    reference_patterns = [
        r"^\(?\d{4}\)?\.",
        r"^[a-z]+,\s*[a-z].*\(\d{4}\)",
        r"\bvol\.\s*\d+",
        r"\bvolume\s+\d+",
        r"\bpp?\.\s*\d+",
        r"\bpages?\s+\d+",
        r"\bjournal\b.*\b\d{4}\b",
    ]

    matches = sum(1 for pattern in reference_patterns if re.search(pattern, text))
    if matches >= 2 and len(text.split()) < 100:
        return True, "reference_like"

    return False, None


RESEARCH_PATTERNS = [
    r"\byield gap\b",
    r"\bregression analysis\b",
    r"\bcorrelation analysis\b",
    r"\btime series\b",
    r"\bstatistical analysis\b",
    r"\btrend analysis\b",
    r"\bp[- ]?value\b",
    r"\bsignificance level\b",
]


def is_research_only(chunk: dict):
    text = normalize_text(chunk.get("text", ""))
    section = normalize_text(chunk.get("section", ""))
    combined = f"{section} {text}"

    matches = sum(1 for pattern in RESEARCH_PATTERNS if re.search(pattern, combined))
    agriculture_score_val, _ = agricultural_score(chunk)

    if matches >= 2 and agriculture_score_val < 2:
        return True, "research_content"

    return False, None


GENERAL_SECTIONS = [
    "general",
    "introduction",
    "background",
]

GENERIC_PATTERNS = [
    r"\bpopulation\b",
    r"\bnatural resources\b",
    r"\beconom",
    r"\bgdp\b",
    r"\bindustr",
    r"\bfuture demand\b",
    r"\bfood security\b",
    r"\bcommercial importance\b",
    r"\bnational importance\b",
    r"\bglobal importance\b",
]


def is_generic_introduction(chunk: dict):
    section = normalize_text(chunk.get("section", ""))
    text = normalize_text(chunk.get("text", ""))

    is_general_section = any(name in section for name in GENERAL_SECTIONS)
    if not is_general_section:
        return False, None

    agriculture_score_val, _ = agricultural_score(chunk)
    if agriculture_score_val >= 2:
        return False, None

    generic_matches = sum(1 for pattern in GENERIC_PATTERNS if re.search(pattern, text))
    if generic_matches >= 2 and agriculture_score_val == 0:
        return True, "generic_introduction"

    return False, None


def evaluate_chunk(chunk: dict):
    """Run all filtering checks on a chunk to decide if it should be excluded from SFT."""
    checks = [
        is_administrative_section,
        contains_metadata,
        is_table_of_contents,
        is_reference_like,
        is_research_only,
        is_heading_only,
        is_low_information,
        is_generic_introduction,
    ]

    for check in checks:
        remove, reason = check(chunk)
        if remove:
            return True, reason

    score, _ = agricultural_score(chunk)
    text = normalize_text(chunk.get("text", ""))
    words = text.split()

    # Strong / moderate agricultural content
    if score >= 4 or score >= 2:
        return False, None

    # Longer factual paragraph with agricultural signal
    if score >= 1 and len(words) >= 30:
        return False, None

    return False, None


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINAL SFT CANDIDATE FILTER")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\nRun classify_chunks.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("sft_candidates.json must contain a JSON list.")

    print(f"Input SFT candidates : {len(chunks)}")

    filtered_chunks = []
    removed_chunks = []
    reason_counts = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            removed_chunks.append({"reason": "invalid_chunk", "chunk": chunk})
            reason_counts["invalid_chunk"] = reason_counts.get("invalid_chunk", 0) + 1
            continue

        remove, reason = evaluate_chunk(chunk)
        if remove:
            removed_chunks.append({"reason": reason, "chunk": chunk})
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            filtered_chunks.append(chunk)

    # Save filtered candidates
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_chunks, f, ensure_ascii=False, indent=2)

    # Save removed candidates
    with open(REMOVED_FILE, "w", encoding="utf-8") as f:
        json.dump(removed_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("FILTERING COMPLETED")
    print("=" * 70)
    print(f"Original candidates : {len(chunks)}")
    print(f"Kept candidates     : {len(filtered_chunks)}")
    print(f"Removed candidates  : {len(removed_chunks)}\n")

    print("Removal reasons:")
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason:<25} : {count}")
    else:
        print("  None")

    print("\nSaved files:")
    print(f"  {OUTPUT_FILE}")
    print(f"  {REMOVED_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
