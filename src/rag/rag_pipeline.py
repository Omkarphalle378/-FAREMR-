"""
=============================================================================
AI-Based Sugarcane Farmer Queries & Support System - Hybrid RAG Pipeline
=============================================================================

Architecture:
1. Multilingual Query Understanding (Marathi Devanagari, English, Roman Marathi, Code-Mixed)
2. Controlled Sugarcane Agricultural Concept & Intent Detection
3. Dual Retrieval:
   - BGE-M3 Semantic Search (FAISS inner product / cosine similarity)
   - Corpus-Wide Concept & Action Scan across all 922 metadata records
4. Intent-Aware Hybrid Ranking (Semantic dominant + Concept + Intent alignment)
5. Grounded Gemini Generation (Strictly factual, preserving context, units, and regional differences)
=============================================================================
"""

import os
import sys
import re
import time
import json
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Ensure Windows terminal handles UTF-8 / Marathi Devanagari output properly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError


# =============================================================================
# CONFIGURATION & PATH RESOLUTION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "data" / "interim" / "faiss" / "sugarcane_faiss.index"
METADATA_FILE = PROJECT_ROOT / "data" / "interim" / "faiss" / "sugarcane_metadata.json"
ENV_FILE = PROJECT_ROOT / ".env"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Gemini model fallback pool across different model generations to prevent quota throttling
MODEL_POOL = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
]

FAISS_CANDIDATE_K = 25
FINAL_TOP_K = 5
MIN_RETRIEVAL_THRESHOLD = 0.30


# =============================================================================
# CONTROLLED AGRICULTURAL CONCEPT TERMINOLOGY MAPPING
# =============================================================================
# Stalk borer, shoot borer, top borer, and root borer are kept strictly separate
# as required by agricultural best practices.

CONCEPTS_MAP = {
    "stalk_borer": {
        "display_name": "Stalk Borer (खोडकीड)",
        "query_terms": [
            "खोडकीड", "खोड किड", "खोड पोखरणारी कीड", "खोड पोखरणारी किड", "खोडातील कीड",
            "khodkid", "khod kid", "khodkida", "khod kida",
            "stalk borer", "stalk-borer", "stalkborer", "stem borer"
        ],
        "corpus_terms": [
            "stalk borer", "stem borer", "chilo", "stalk-borer", "stalkborer",
            "trichogramma", "cotesia", "thimet", "chlorpyriphos"
        ]
    },
    "shoot_borer": {
        "display_name": "Shoot Borer (शूट बोरर / खोडकीड)",
        "query_terms": [
            "शूट बोरर", "शूटबोरर", "अर्ली शूट बोरर",
            "shoot borer", "shoot-borer", "shootborer", "early shoot borer"
        ],
        "corpus_terms": [
            "shoot borer", "early shoot borer", "chilo infuscatellus"
        ]
    },
    "top_borer": {
        "display_name": "Top Borer (शेंडा कीड)",
        "query_terms": [
            "शेंडा कीड", "शेंडा किड", "शेंडा पोखरणारी कीड", "शेंडा पोखरणारी किड",
            "shenda kid", "shendakid", "shenda kida",
            "top borer", "top-borer", "topborer"
        ],
        "corpus_terms": [
            "top borer", "scirpophaga", "bunchy top"
        ]
    },
    "root_borer": {
        "display_name": "Root Borer (मुळातील कीड)",
        "query_terms": [
            "मुळातील कीड", "मुळाची कीड", "mula kid", "mulatil kid",
            "root borer", "root-borer", "rootborer"
        ],
        "corpus_terms": [
            "root borer", "emmalocera depressella"
        ]
    },
    "white_grub": {
        "display_name": "White Grub (हुमणी)",
        "query_terms": [
            "हुमणी", "हुमणी कीड", "humani", "humani kid", "white grub", "white-grub"
        ],
        "corpus_terms": [
            "white grub", "holotrichia", "light trap", "phorate"
        ]
    },
    "water_requirement": {
        "display_name": "Water Requirement (पाण्याची गरज)",
        "query_terms": [
            "पाणी किती", "पाण्याची गरज", "पाणी लागते", "पाणी किती लागते", "पावसाचे प्रमाण",
            "pani kiti", "panyachi garaj", "pani lagte", "water requirement", "water needs", "water demand"
        ],
        "corpus_terms": [
            "water requirement", "quantity of water", "1500 to 2500 mm", "1400-1500 mm",
            "1850 to 2500 mm", "consumes about 250 tons", "60-70 tonnes of water", "water requirements"
        ]
    },
    "irrigation": {
        "display_name": "Irrigation Management (सिंचन / पाणी व्यवस्थापन)",
        "query_terms": [
            "सिंचन", "पाण्याचे नियोजन", "पाणी व्यवस्थापन", "पाणी कधी द्यावे", "पाणी किती वेळाने",
            "sinchan", "panyache niyojan", "irrigation", "irrigation schedule", "irrigation interval", "furrow irrigation"
        ],
        "corpus_terms": [
            "irrigation water management", "irrigation interval", "days interval", "irrigate", "frequency of irrigation"
        ]
    },
    "drip": {
        "display_name": "Drip Irrigation (ठिबक सिंचन)",
        "query_terms": [
            "ठिबक", "ठिबक सिंचन", "thibak", "thibak sinchan", "drip", "drip irrigation", "fertigation"
        ],
        "corpus_terms": [
            "drip irrigation", "fertigation", "sub-surface drip", "water saving"
        ]
    },
    "fertilizer": {
        "display_name": "Fertilizer / Nutrient (खत व्यवस्थापन)",
        "query_terms": [
            "खत", "खते", "खताची मात्रा", "खत कधी द्यावे", "खत किती द्यावे", "नत्र", "स्फुरद", "पालाश", "युरिया",
            "khat", "khate", "fertilizer", "fertiliser", "npk", "urea", "nitrogen", "phosphorus", "potassium", "dap"
        ],
        "corpus_terms": [
            "fertilizer", "fertiliser", "nutrient management", "nitrogen", "phosphorus", "potassium",
            "npk", "urea", "top dressing", "zinc sulfate", "ferrous sulphate", "calcareous soils"
        ]
    },
    "weed": {
        "display_name": "Weed Control (तण नियंत्रण)",
        "query_terms": [
            "तण", "तण नियंत्रण", "तणनाशक", "खुरपणी", "तण कसे नियंत्रित",
            "tan", "tan niyantran", "khurpani", "weed", "weeds", "weed control", "weed management", "herbicide"
        ],
        "corpus_terms": [
            "weed", "weeds", "weed management", "weed control", "atrazine", "herbicide", "intercultural operations", "hoeing"
        ]
    },
    "planting": {
        "display_name": "Planting (लागवड)",
        "query_terms": [
            "लागवड", "लागवडीची पद्धत", "लागवड कशी करावी", "रोपे", "बेणे",
            "lagwad", "lagwad kashi karavi", "planting", "sowing", "setts", "trench planting"
        ],
        "corpus_terms": [
            "planting", "trench method", "furrow method", "autumn planted", "spring planted", "cane setts"
        ]
    },
    "spacing": {
        "display_name": "Spacing (लागवडीचे अंतर)",
        "query_terms": [
            "अंतर", "अंतरावर", "लागवडीचे अंतर", "दोन ओळीतील अंतर",
            "antar", "antar kiti", "spacing", "row spacing", "distance", "paired row"
        ],
        "corpus_terms": [
            "spacing", "row to row", "paired row", "90 cm", "120 cm", "150 cm", "distance between rows"
        ]
    },
    "seed_cane": {
        "display_name": "Seed Cane / Sett Treatment (बेणे प्रक्रिया)",
        "query_terms": [
            "बेणे", "बेणे प्रक्रिया", "बीज प्रक्रिया", "bene", "bene prakriya", "setts", "sett treatment", "bud chip"
        ],
        "corpus_terms": [
            "setts", "seed cane", "bud chip", "sett treatment", "carbendazim", "hot water treatment"
        ]
    },
    "disease": {
        "display_name": "Disease Management (रोग नियंत्रण)",
        "query_terms": [
            "रोग", "रोगाची लक्षणे", "रोग नियंत्रण", "rog", "rog niyantran", "disease", "diseases"
        ],
        "corpus_terms": [
            "disease", "diseases", "fungus", "red rot", "smut", "wilt", "rust", "pathogen"
        ]
    },
    "rust": {
        "display_name": "Rust (तांबेरा)",
        "query_terms": [
            "तांबेरा", "तांबेरा रोग", "tambera", "tambera rog", "rust", "rust disease"
        ],
        "corpus_terms": [
            "rust", "puccinia melanocephala", "brown spots", "urediniospores"
        ]
    },
    "smut": {
        "display_name": "Smut (काणी)",
        "query_terms": [
            "काणी", "काणी रोग", "चाबूक काणी", "kani", "kani rog", "smut", "smut disease"
        ],
        "corpus_terms": [
            "smut", "sporisorium scitamineum", "whip smut", "black whip"
        ]
    },
    "wilt": {
        "display_name": "Wilt (मर रोग)",
        "query_terms": [
            "मर", "मर रोग", "mar", "mar rog", "wilt", "wilt disease"
        ],
        "corpus_terms": [
            "wilt", "fusarium", "cephalosporium sacchari"
        ]
    },
    "ratoon": {
        "display_name": "Ratoon Management (खोडवा व्यवस्थापन)",
        "query_terms": [
            "खोडवा", "खोडव्याचे नियोजन", "खोडवा पीक", "khodwa", "khodva", "ratoon", "ratooning", "ratoon management"
        ],
        "corpus_terms": [
            "ratoon", "ratooning", "ratoon management", "stubble shaving", "gap filling", "trash mulching"
        ]
    },
    "harvesting": {
        "display_name": "Harvesting & Maturity (काढणी व परिपक्वता)",
        "query_terms": [
            "काढणी", "तोडणी", "कापणी", "परिपक्वता", "kadhani", "todani", "harvest", "harvesting", "maturity", "brix"
        ],
        "corpus_terms": [
            "harvest", "harvesting", "maturity", "sucrose content", "hand refractometer", "brix"
        ]
    },
    "trash_management": {
        "display_name": "Trash Mulching (पाचट व्यवस्थापन)",
        "query_terms": [
            "पाचट", "पाचट व्यवस्थापन", "कचरा आच्छादन", "pachat", "pachat vyavasthapan", "trash mulching", "mulch"
        ],
        "corpus_terms": [
            "trash mulching", "trash mulching in sugarcane", "conserve soil moisture", "press mud"
        ]
    }
}


# =============================================================================
# QUERY INTENT DEFINITIONS & ACTIONABLE EVIDENCE TERMS
# =============================================================================

INTENTS_MAP = {
    "control": {
        "query_signals": [
            "नियंत्रित", "नियंत्रण", "उपाय", "व्यवस्थापन", "कशी करावी", "कसा करावा", "कसे करावे",
            "बंदोबस्त", "रोकथाम", "औषध", "फवारणी",
            "niyantrit", "niyantran", "upay", "vyavasthapan", "kashi karavi", "kasa karava",
            "kase karave", "favarni", "control", "manage", "management", "treatment",
            "prevent", "prevention", "eradicate"
        ],
        # Chunks containing active management methods score high
        "action_evidence": [
            "control", "management", "application of", "spray", "spraying", "kg/ha", "litres/ha",
            "ml/litre", "trichogramma", "cotesia", "thimet", "carbofuran", "chlorpyriphos",
            "pheromone trap", "light trap", "collection and destruction", "chemical measures",
            "biological measures", "cultural practices"
        ],
        # Penalty for passive resistance lists that don't provide management advice
        "passive_evidence": [
            "resistant to", "moderately resistant", "less susceptible"
        ]
    },
    "requirement": {
        "query_signals": [
            "किती", "गरज", "प्रमाण", "मात्रा", "डोस", "लागते", "आवश्यकता",
            "kiti", "garaj", "praman", "matra", "dose", "lagte",
            "how much", "how many", "requirement", "requirements", "quantity", "need", "dose"
        ],
        "action_evidence": [
            "requirement", "requirements", "water requirement", "annual water requirement",
            "1500 to 2500 mm", "1400-1500 mm", "1850 to 2500 mm", "kg/ha", "t/ha", "tonnes",
            "consumes about 250 tons", "60-70 tonnes of water"
        ],
        "passive_evidence": []
    },
    "timing": {
        "query_signals": [
            "कधी", "वेळ", "वेळापत्रक", "हप्ते", "टप्पे", "किती दिवसांनी",
            "kadhi", "vel", "velapatrak", "hapte", "tappe", "kiti divasani",
            "when", "timing", "schedule", "interval", "splits", "stage", "frequency"
        ],
        "action_evidence": [
            "days after planting", "weeks after planting", "equal splits", "split application",
            "basal dose", "top dressing", "first top dressing", "irrigation interval", "days interval"
        ],
        "passive_evidence": []
    },
    "spacing": {
        "query_signals": [
            "अंतर", "अंतरावर", "अंतर किती", "दोन ओळीतील",
            "antar", "antaravar", "antar kiti", "spacing", "distance", "row spacing", "how far", "apart"
        ],
        "action_evidence": [
            "spacing", "row to row", "paired row", "trench", "distance", "90 cm", "120 cm",
            "150 cm", "paired row sugarcane", "inter-row"
        ],
        "passive_evidence": []
    },
    "symptoms": {
        "query_signals": [
            "ओळख", "कशी ओळखावी", "लक्षणे", "स्वरुप", "नुकसान",
            "olakh", "kashi olakhaychi", "lakshane", "symptoms", "signs", "identify", "identification", "damage"
        ],
        "action_evidence": [
            "symptoms", "nature of damage", "abnormal proliferation", "yellowing", "wilting",
            "dead heart", "tunneling", "burrowing", "whip"
        ],
        "passive_evidence": []
    },
    "variety": {
        "query_signals": [
            "वाण", "जात", "जाती", "बेस्ट जात",
            "wan", "jat", "jati", "variety", "varieties", "cultivar", "best variety"
        ],
        "action_evidence": [
            "variety", "varieties", "cultivar", "co 86032", "cop", "coc", "sugar yield", "cane yield"
        ],
        "passive_evidence": []
    }
}


# =============================================================================
# ENVIRONMENT & CLIENT INITIALIZATION
# =============================================================================

load_dotenv(ENV_FILE, override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        f"GEMINI_API_KEY not found in {ENV_FILE}. Please provide a valid Gemini API key."
    )

client = genai.Client(api_key=api_key)


# =============================================================================
# LOAD EMBEDDING MODEL & FAISS INDEX
# =============================================================================

print("=" * 60)
print("INITIALIZING SUGARCANE HYBRID RAG SYSTEM")
print("=" * 60)

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded successfully.")

print(f"\nLoading FAISS index from: {INDEX_FILE}")
if not INDEX_FILE.exists():
    raise FileNotFoundError(f"FAISS index file missing: {INDEX_FILE}")

faiss_index = faiss.read_index(str(INDEX_FILE))
print(f"FAISS total indexed vectors: {faiss_index.ntotal}")

print(f"\nLoading metadata from: {METADATA_FILE}")
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Metadata file missing: {METADATA_FILE}")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    corpus_metadata = json.load(f)

print(f"Metadata total records: {len(corpus_metadata)}")

if faiss_index.ntotal != len(corpus_metadata):
    raise ValueError(
        f"Mismatch: FAISS vectors ({faiss_index.ntotal}) != metadata records ({len(corpus_metadata)})"
    )

print("System initialized successfully.\n" + "=" * 60)


# =============================================================================
# QUERY UNDERSTANDING (CONCEPTS & INTENT)
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation for consistent matching."""
    text = str(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_query_concepts(query: str) -> list:
    """
    Detects agricultural concepts from Marathi, Roman Marathi, or English query.
    Keeps pests (stalk borer, shoot borer, top borer, root borer) distinct.
    """
    norm_q = normalize_text(query)
    detected = []

    for concept_id, spec in CONCEPTS_MAP.items():
        for term in spec["query_terms"]:
            pattern = r"\b" + re.escape(term) + r"\b" if term.isascii() else re.escape(term)
            if re.search(pattern, norm_q, re.IGNORECASE):
                if concept_id not in detected:
                    detected.append(concept_id)
                break

    # Contextual priority: if specific borer is detected (e.g. stalk borer),
    # avoid generic fallback to shoot borer unless requested.
    return detected


def detect_query_intent(query: str) -> str:
    """
    Detects user agricultural intent (control, requirement, timing, spacing, symptoms, etc.).
    """
    norm_q = normalize_text(query)

    for intent_name, intent_spec in INTENTS_MAP.items():
        for sig in intent_spec["query_signals"]:
            pattern = r"\b" + re.escape(sig) + r"\b" if sig.isascii() else re.escape(sig)
            if re.search(pattern, norm_q, re.IGNORECASE):
                return intent_name

    return "general"


def reformulate_query(query: str, detected_concepts: list, detected_intent: str) -> str:
    """
    Generates an internal concise English search phrase for retrieval.
    Has a deterministic local fallback based on concepts and intents,
    so retrieval NEVER breaks if Gemini API is unreachable or rate-limited.
    """
    norm_q = normalize_text(query)
    is_pure_english = norm_q.isascii() and not any(
        w in norm_q for w in ["usala", "oos", "pikala", "khat", "pani", "kiti", "dyach", "khodkid", "kashi", "niyantrit"]
    )

    if is_pure_english:
        return query.strip()

    # Deterministic local mapping (Zero dependency on external API)
    concept_keywords = {
        "stalk_borer": "sugarcane stalk borer",
        "shoot_borer": "sugarcane shoot borer",
        "top_borer": "sugarcane top borer",
        "root_borer": "sugarcane root borer",
        "white_grub": "sugarcane white grub",
        "water_requirement": "sugarcane water requirement",
        "irrigation": "sugarcane irrigation management interval",
        "drip": "sugarcane drip irrigation fertigation",
        "fertilizer": "sugarcane fertilizer application NPK dose",
        "weed": "sugarcane weed control herbicide",
        "planting": "sugarcane planting method",
        "spacing": "sugarcane planting row spacing distance",
        "seed_cane": "sugarcane sett treatment seed cane",
        "disease": "sugarcane disease management",
        "rust": "sugarcane rust disease symptoms control",
        "smut": "sugarcane smut disease control",
        "wilt": "sugarcane wilt disease management",
        "ratoon": "sugarcane ratoon crop management",
        "harvesting": "sugarcane harvesting maturity",
        "trash_management": "sugarcane trash mulching management"
    }

    intent_keywords = {
        "control": "control management recommendations",
        "requirement": "requirement quantity",
        "timing": "application timing schedule splits",
        "spacing": "spacing distance between rows",
        "symptoms": "symptoms identification damage",
        "variety": "varieties recommendations"
    }

    # Build local deterministic search phrase
    local_terms = []
    if detected_concepts:
        for c in detected_concepts:
            if c in concept_keywords:
                local_terms.append(concept_keywords[c])
    else:
        local_terms.append("sugarcane")

    if detected_intent in intent_keywords:
        local_terms.append(intent_keywords[detected_intent])

    local_fallback_query = " ".join(local_terms)

    # Optional fast enhancement via Gemini with instant timeout safety
    prompt = (
        f"You are a sugarcane retrieval optimizer. Convert this farmer query into 3-5 concise English search words:\n"
        f"Query: '{query}'\nConcepts: {detected_concepts}, Intent: {detected_intent}\n"
        f"Return ONLY the concise English search words:"
    )

    for model_name in MODEL_POOL[:2]:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=30
                )
            )
            if resp.text:
                cleaned = resp.text.strip().replace('"', '').replace("'", "")
                if len(cleaned.split()) >= 2:
                    return cleaned
        except Exception:
            continue

    return local_fallback_query


# =============================================================================
# CORPUS-WIDE CONCEPT SCAN (Scans all 922 chunks for high recall)
# =============================================================================

def scan_corpus_for_concepts(detected_concepts: list) -> list:
    """
    Scans ALL 922 metadata records for the detected concept terms.
    Ensures that chunks like chunk_000153 (Section: borer, trichogramma, chlorpyriphos)
    enter the candidate pool even if FAISS initial top-K ranked them lower.
    """
    if not detected_concepts:
        return []

    matched_indices = []

    # Gather all target corpus terms for the detected concepts
    target_corpus_terms = []
    for cid in detected_concepts:
        if cid in CONCEPTS_MAP:
            target_corpus_terms.extend(CONCEPTS_MAP[cid]["corpus_terms"])

    for idx, item in enumerate(corpus_metadata):
        section_text = (item.get("section", "") + " " + item.get("text", "")).lower()

        # Check for any term match
        for term in target_corpus_terms:
            if term.lower() in section_text:
                matched_indices.append(idx)
                break

    return matched_indices


# =============================================================================
# INTENT-AWARE HYBRID RANKING & SCORING
# =============================================================================

def score_candidate(
    chunk: dict,
    chunk_idx: int,
    query_vector: np.ndarray,
    detected_concepts: list,
    detected_intent: str,
    raw_query: str
) -> dict:
    """
    Computes a transparent, reproducible hybrid relevance score:
    - semantic_score (50%): BGE-M3 cosine similarity (dominant baseline)
    - concept_score  (25%): Exact presence of target concept terms in chunk
    - intent_score   (20%): Actionable management evidence matching query intent
    - phrase_score   (05%): Direct key phrase presence
    """
    # 1. Semantic Similarity from FAISS vector
    chunk_vector = faiss_index.reconstruct(int(chunk_idx))
    semantic_score = float(np.dot(query_vector[0], chunk_vector))

    full_text = (chunk.get("section", "") + " " + chunk.get("text", "")).lower()

    # 2. Concept Match Score
    concept_score = 0.0
    if detected_concepts:
        primary_concept = detected_concepts[0]
        if primary_concept in CONCEPTS_MAP:
            terms = CONCEPTS_MAP[primary_concept]["corpus_terms"]
            matched_terms = [t for t in terms if t.lower() in full_text]
            if matched_terms:
                # Strong concept presence
                concept_score = min(0.6 + 0.1 * len(matched_terms), 1.0)
            else:
                concept_score = 0.0
    else:
        concept_score = 0.5  # Neutral when no specific concept is isolated

    # 3. Intent Match Score (Actionable Recommendations vs Passive Lists)
    intent_score = 0.0
    if detected_intent in INTENTS_MAP:
        intent_spec = INTENTS_MAP[detected_intent]
        action_matches = [t for t in intent_spec["action_evidence"] if t.lower() in full_text]
        passive_matches = [t for t in intent_spec["passive_evidence"] if t.lower() in full_text]

        if action_matches:
            # Chunk contains concrete actionable advice (e.g. spray doses, trichogramma, mm requirements)
            intent_score = min(0.6 + 0.1 * len(action_matches), 1.0)
        elif passive_matches:
            # Chunk merely mentions 'resistant to...' without management recommendations
            intent_score = 0.15
        else:
            intent_score = 0.0
    else:
        intent_score = 0.5  # Neutral for general intents

    # Section relevance boost
    section_boost = 0.0
    sec_lower = chunk.get("section", "").lower()
    if detected_concepts:
        for c in detected_concepts:
            if c in sec_lower or sec_lower in c:
                section_boost = 0.05
                break

    # 4. Phrase Match Score
    phrase_score = 0.0
    raw_words = [w for w in raw_query.lower().split() if len(w) >= 4 and w.isascii()]
    if raw_words:
        matched_words = [w for w in raw_words if w in full_text]
        phrase_score = len(matched_words) / len(raw_words)

    # Combined Final Score
    # Semantic remains the dominant signal (0.50).
    # Concept & Intent ensure actionable domain chunks (like chunk_000153)
    # outrank passive variety resistance tables (like chunk_000051).
    final_score = (
        0.50 * semantic_score +
        0.25 * concept_score +
        0.20 * intent_score +
        0.05 * phrase_score +
        section_boost
    )

    result = chunk.copy()
    result["chunk_index"] = chunk_idx
    result["semantic_score"] = float(semantic_score)
    result["concept_score"] = float(concept_score)
    result["intent_score"] = float(intent_score)
    result["phrase_score"] = float(phrase_score)
    result["final_score"] = float(final_score)

    return result


# =============================================================================
# HYBRID RETRIEVAL PIPELINE
# =============================================================================

def retrieve_hybrid_chunks(query: str, top_k: int = FINAL_TOP_K):
    """
    Executes the full hybrid retrieval pipeline:
    1. Detect Concepts & Intent
    2. Reformulate English search phrase
    3. Query FAISS for top-K candidates
    4. Scan all 922 metadata records for concept matches
    5. Merge candidates into unified candidate pool
    6. Intent-aware hybrid scoring & ranking
    7. Select Top 5 relevant chunks
    """
    detected_concepts = detect_query_concepts(query)
    detected_intent = detect_query_intent(query)
    search_query = reformulate_query(query, detected_concepts, detected_intent)

    # Embed search query with BGE-M3
    query_vector = embedding_model.encode([search_query], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    # Step A: FAISS Semantic Search
    scores, indices = faiss_index.search(query_vector, FAISS_CANDIDATE_K)

    candidate_indices = set()
    for idx in indices[0]:
        if idx != -1:
            candidate_indices.add(int(idx))

    # Step B: Corpus-Wide Concept Scan across ALL 922 chunks
    concept_matched_indices = scan_corpus_for_concepts(detected_concepts)
    for idx in concept_matched_indices:
        candidate_indices.add(int(idx))

    # Step C: Intent-Aware Hybrid Scoring on the Unified Candidate Pool
    scored_candidates = []
    for idx in candidate_indices:
        chunk = corpus_metadata[idx]
        scored_chunk = score_candidate(
            chunk=chunk,
            chunk_idx=idx,
            query_vector=query_vector,
            detected_concepts=detected_concepts,
            detected_intent=detected_intent,
            raw_query=query
        )
        scored_candidates.append(scored_chunk)

    # Sort descending by final_score
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Step D: Select Top K ensuring diversity and quality
    selected_chunks = []
    seen_ids = set()

    for c in scored_candidates:
        cid = c.get("id")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        # Retain if it passes minimum threshold
        if c["final_score"] >= MIN_RETRIEVAL_THRESHOLD:
            selected_chunks.append(c)

        if len(selected_chunks) >= top_k:
            break

    # Fallback to top scored if strict threshold left no candidates
    if not selected_chunks and scored_candidates:
        selected_chunks.append(scored_candidates[0])

    return selected_chunks, search_query, detected_concepts, detected_intent


# =============================================================================
# CONTEXT COMPOSER
# =============================================================================

def build_context(chunks: list) -> str:
    """Formats retrieved chunks with strict metadata tracing."""
    parts = []
    for num, chunk in enumerate(chunks, start=1):
        parts.append(
            f"SOURCE {num} [Chunk ID: {chunk.get('id', 'N/A')} | Source: {chunk.get('source', 'Unknown')} | Page: {chunk.get('page', 'N/A')} | Section: {chunk.get('section', 'General')}]\n"
            f"{chunk.get('text', '').strip()}"
        )
    return "\n\n" + ("\n" + "=" * 60 + "\n").join(parts)


# =============================================================================
# GROUNDED GEMINI ANSWER GENERATION
# =============================================================================

def generate_grounded_answer(
    question: str,
    chunks: list,
    detected_concepts: list,
    detected_intent: str,
    search_query: str
) -> str:
    """
    Generates farmer answer strictly grounded in retrieved chunks.
    Preserves numerical values, units, regional differences, and refuses safely if irrelevant.
    """
    fallback_marathi = "उपलब्ध स्रोतांमध्ये या प्रश्नाचे पुरेसे उत्तर मिळाले नाही."
    fallback_english = "Sufficient information to answer this question is not available in the provided sources."

    # Determine user language
    has_devanagari = any("\u0900" <= c <= "\u097F" for c in question)
    words_lower = set(re.findall(r"[a-z]+", question.lower()))
    roman_indicators = {
        "usala", "oos", "pikala", "khat", "pani", "kiti", "dyach", "dyaycha", "dyave",
        "khodkid", "kashi", "niyantrit", "karavi", "karaycha", "upay", "rog", "tan", "antor", "antar"
    }
    is_marathi = has_devanagari or bool(words_lower & roman_indicators)
    fallback_msg = fallback_marathi if is_marathi else fallback_english

    # Safety Guardrail: If no chunks or top chunk has zero relevance to detected concept
    if not chunks:
        return fallback_msg

    top_chunk = chunks[0]
    if detected_concepts and top_chunk.get("concept_score", 0.0) == 0.0 and top_chunk.get("semantic_score", 0.0) < 0.40:
        return fallback_msg

    context = build_context(chunks)

    concept_names = [CONCEPTS_MAP[c]["display_name"] for c in detected_concepts if c in CONCEPTS_MAP]
    concept_note = ""
    if concept_names:
        concept_note = f"\nAGRICULTURAL CONTEXT: The farmer's question relates to {', '.join(concept_names)}. Match this to the corresponding English terms in the source content (e.g., खोडकीड / khodkid = stalk borer / shoot borer, पाणी = irrigation / water, खत = fertilizer / nutrients, तण = weed control, लागवड अंतर = spacing / planting geometry)."

    prompt = f"""You are a trusted sugarcane agricultural assistant for farmers.

Answer the farmer's question using ONLY the supplied SOURCE CONTENT.{concept_note}

FARMER QUESTION:
{question}
(Target Search Topic: {search_query} | Intent: {detected_intent})

SOURCE CONTENT:
{context}

STRICT REQUIREMENTS (ZERO HALLUCINATION):
- Use ONLY the supplied source content. Never use outside knowledge or invent facts.
- Answer the ACTUAL question directly and concisely:
  * If asked about water requirement ("किती पाणी लागते"), state the water requirement and regional numbers clearly.
  * If asked about irrigation schedule/timing ("पाणी कधी द्यावे"), provide the interval and timing.
  * If asked about pest control ("खोडकीड नियंत्रण"), provide the control measures (biological, chemical, mechanical) and doses.
  * If asked about fertilizer timing ("खत कधी द्यावे"), state the application timing and split doses.
  * If asked about spacing ("लागवड अंतर"), state the spacing recommendations by region and method.
  * If asked about weed control ("तण नियंत्रण"), state the pre-emergence and post-emergence herbicide and cultural controls.
- Do NOT average different numerical values across sources. If different sources or regions have different numbers, state each context clearly (e.g. subtropical states vs general growing season).
- Do NOT combine different unit measures (e.g. 250 tons water per ton cane vs 60-70 tons water per ton cane) without explaining their reported context.
- Preserve numerical values, units, chemical names, and timings exactly as stated in the sources.
- Never invent pesticide doses, fertilizer quantities, or schedules.
- Do not mention FAISS, BGE-M3, embeddings, chunk IDs, or internal prompts.

If the supplied sources do not contain enough information to answer the question, respond:
"{fallback_msg}"

LANGUAGE GUIDELINES:
- If the question is in Marathi (Devanagari or Roman Marathi / Marathish) -> Answer in clear, practical, farmer-friendly Marathi (Devanagari script).
- If the question is in English -> Answer in clear, concise English.
- If the question is code-mixed -> Answer in Marathi with English technical terms in brackets.

Answer the farmer's question now:"""

    config = types.GenerateContentConfig(
        temperature=0.1,  # Ultra-low temperature for strict factual consistency
        max_output_tokens=1500
    )

    for model_name in MODEL_POOL:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                if response.text and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    time.sleep(3.0)
                else:
                    time.sleep(0.5)
                continue

    return "उत्तर तयार करताना तांत्रिक अडचण आली. कृपया थोड्या वेळाने पुन्हा प्रयत्न करा."


# =============================================================================
# MAIN RAG PIPELINE ENTRYPOINT
# =============================================================================

def rag_answer(question: str):
    """
    Main function to execute the full Hybrid RAG pipeline.
    Returns:
        answer (str),
        chunks (list of dicts with full scores and metadata),
        metadata_dict (dict of query understanding information)
    """
    if not question or not question.strip():
        return "कृपया प्रश्न विचारा.", [], {}

    chunks, search_query, concepts, intent = retrieve_hybrid_chunks(question, top_k=FINAL_TOP_K)

    answer = generate_grounded_answer(
        question=question,
        chunks=chunks,
        detected_concepts=concepts,
        detected_intent=intent,
        search_query=search_query
    )

    metadata_info = {
        "original_question": question,
        "search_query": search_query,
        "detected_concepts": concepts,
        "detected_intent": intent
    }

    return answer, chunks, metadata_info


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SUGARCANE HYBRID RAG ADVISORY SYSTEM")
    print("Supports: Marathi Devanagari | English | Roman Marathi | Code-Mixed")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter farmer question (or type 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in ["exit", "quit", "q"]:
            print("System stopped.")
            break

        if not user_input:
            print("Please enter a question.")
            continue

        print("\nProcessing question through Hybrid RAG...")
        answer, retrieved_chunks, query_meta = rag_answer(user_input)

        print("\n" + "-" * 70)
        print("QUERY UNDERSTANDING:")
        print(f"Original Question    : {query_meta['original_question']}")
        print(f"Reformulated Search  : {query_meta['search_query']}")
        print(f"Detected Concepts    : {query_meta['detected_concepts']}")
        print(f"Detected Intent      : {query_meta['detected_intent']}")
        print("-" * 70)

        print(f"\nTOP RETRIEVED SOURCES ({len(retrieved_chunks)} chunks):")
        print("=" * 70)

        for i, c in enumerate(retrieved_chunks, start=1):
            print(f"Rank {i} | Chunk ID: {c.get('id')} | Page: {c.get('page')} | Section: {c.get('section')}")
            print(f"       Semantic: {c.get('semantic_score', 0):.4f} | Concept: {c.get('concept_score', 0):.4f} | Intent: {c.get('intent_score', 0):.4f} | Final: {c.get('final_score', 0):.4f}")
            print(f"       Source  : {c.get('source')}")
            print(f"       Content : {c.get('text', '')[:200].replace(chr(10), ' ')}...")
            print("-" * 70)

        print("\n" + "=" * 70)
        print("GENERATED ANSWER:")
        print("=" * 70)
        print(f"\n{answer}\n")
        print("=" * 70)
