"""Classification of knowledge chunks into RAG references and SFT candidates."""

import json
from pathlib import Path

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "validated" / "final_chunks.json"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "classified"

# Sugarcane topics useful for farmer questions
FARMER_TOPICS = {
    "variety": [
        "variety",
        "varieties",
        "cultivar",
        "sugarcane variety",
        "recommended variety"
    ],
    "planting": [
        "planting",
        "planting method",
        "planting material",
        "planting time",
        "setts",
        "sett",
        "seed cane",
        "seed material",
        "sett treatment",
        "germination"
    ],
    "soil": [
        "soil",
        "soil preparation",
        "land preparation",
        "tillage",
        "soil fertility",
        "soil health"
    ],
    "fertilizer": [
        "fertilizer",
        "fertiliser",
        "fertilization",
        "fertilisation",
        "nitrogen",
        "phosphorus",
        "potassium",
        "urea",
        "nutrient",
        "nutrients",
        "manure",
        "compost"
    ],
    "irrigation": [
        "irrigation",
        "water management",
        "water requirement",
        "water requirement of sugarcane",
        "drip irrigation",
        "irrigation schedule"
    ],
    "weed_management": [
        "weed",
        "weeds",
        "weed management",
        "weed control",
        "herbicide"
    ],
    "pest_management": [
        "pest",
        "pests",
        "pest management",
        "pest control",
        "insect",
        "insects",
        "insect pest"
    ],
    "disease_management": [
        "disease",
        "diseases",
        "disease management",
        "disease control",
        "pathogen"
    ],
    "ratoon": [
        "ratoon",
        "ratooning",
        "ratoon crop",
        "ratoon management"
    ],
    "harvesting": [
        "harvest",
        "harvesting",
        "harvest maturity",
        "maturity",
        "cutting",
        "cane cutting"
    ],
    "trash_management": [
        "trash",
        "trash management",
        "trash mulching",
        "trash blanket",
        "sugarcane trash"
    ],
    "organic_farming": [
        "organic farming",
        "organic sugarcane",
        "organic manure",
        "biofertilizer",
        "biofertiliser",
        "biological"
    ],
    "climate": [
        "climate",
        "climate change",
        "temperature",
        "rainfall",
        "drought",
        "weather",
        "climatic"
    ]
}

# Topics mainly useful as RAG reference material
RESEARCH_TOPICS = [
    "yield gap",
    "yield gap analysis",
    "productivity",
    "growth analysis",
    "growth instability",
    "growth rate",
    "acreage",
    "area under sugarcane",
    "production trend",
    "production analysis",
    "economic analysis",
    "economics",
    "cost of cultivation",
    "profitability",
    "price analysis",
    "statistical analysis",
    "regression",
    "correlation",
    "time series",
    "trend analysis"
]

# Content that should normally be excluded
EXCLUDE_TERMS = [
    "references",
    "bibliography",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "author information",
    "copyright",
    "isbn",
    "issn",
    "www.",
    "http://",
    "https://"
]


def find_topic(section: str, text: str) -> str:
    """Identify matching sugarcane topic based on keyword occurrences."""
    combined = f"{section} {text}".lower()
    topic_scores = {}

    for topic, keywords in FARMER_TOPICS.items():
        score = 0
        for keyword in keywords:
            if keyword in combined:
                score += 1
                # Give extra importance if keyword occurs in section name
                if keyword in section.lower():
                    score += 2

        if score > 0:
            topic_scores[topic] = score

    if not topic_scores:
        return "general"

    return max(topic_scores, key=topic_scores.get)


def should_exclude(section: str, text: str) -> bool:
    """Determine whether chunk contains bibliographic or metadata noise."""
    combined = f"{section} {text}".lower()
    for term in EXCLUDE_TERMS:
        if term in combined:
            return True
    return False


def classify_chunk(chunk: dict) -> dict:
    """Determine chunk classification category (rag_sft, rag_only, exclude)."""
    section = chunk.get("section", "")
    text = chunk.get("text", "")

    # 1. Exclude obvious non-knowledge material
    if should_exclude(section, text):
        return {
            "category": "exclude",
            "topic": "none",
            "reason": "reference_or_metadata"
        }

    # 2. Determine topic
    topic = find_topic(section, text)
    combined = f"{section} {text}".lower()

    # 3. Research/economic information
    research_score = sum(1 for kw in RESEARCH_TOPICS if kw in combined)

    # 4. Farmer-useful information
    farmer_score = 0
    for keywords in FARMER_TOPICS.values():
        for keyword in keywords:
            if keyword in combined:
                farmer_score += 1

    # 5. Classification
    if farmer_score > 0:
        category = "rag_sft"
        reason = "farmer_relevant_agricultural_knowledge"
    elif research_score > 0:
        category = "rag_only"
        reason = "research_or_reference_information"
    else:
        category = "rag_only"
        reason = "potential_reference_information"

    return {
        "category": category,
        "topic": topic,
        "reason": reason
    }


def run_classification(input_path: Path = INPUT_FILE, output_dir: Path = OUTPUT_FOLDER) -> None:
    """Classify knowledge chunks into RAG and SFT candidate sets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOADING FINAL CHUNKS")
    print("=" * 60)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total input chunks: {len(chunks)}")

    rag_chunks = []
    sft_candidates = []
    excluded_chunks = []
    total = 0

    for chunk in chunks:
        total += 1
        result = classify_chunk(chunk)

        record = {
            "id": chunk.get("id"),
            "source": chunk.get("source"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "topic": result["topic"],
            "text": chunk.get("text"),
            "classification_reason": result["reason"]
        }

        # RAG
        if result["category"] in ["rag_sft", "rag_only"]:
            rag_chunks.append(record)

        # SFT candidates
        if result["category"] == "rag_sft":
            sft_candidates.append(record)

        # Excluded
        if result["category"] == "exclude":
            excluded_chunks.append(record)

    # Save results
    with open(output_dir / "rag_chunks.json", "w", encoding="utf-8") as f:
        json.dump(rag_chunks, f, ensure_ascii=False, indent=2)

    with open(output_dir / "sft_candidates.json", "w", encoding="utf-8") as f:
        json.dump(sft_candidates, f, ensure_ascii=False, indent=2)

    with open(output_dir / "excluded.json", "w", encoding="utf-8") as f:
        json.dump(excluded_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total chunks       : {total}")
    print(f"RAG chunks         : {len(rag_chunks)}")
    print(f"SFT candidates     : {len(sft_candidates)}")
    print(f"Excluded chunks    : {len(excluded_chunks)}")
    print("\nCreated:")
    print(f"{output_dir}/")
    print("├── rag_chunks.json")
    print("├── sft_candidates.json")
    print("└── excluded.json")


def main():
    run_classification()


if __name__ == "__main__":
    main()
