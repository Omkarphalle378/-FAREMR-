# AI-Based Farmer Queries & Support System

An AI-powered farmer support system designed to provide reliable, source-grounded agricultural guidance to farmers.

The initial implementation focuses on **sugarcane cultivation in Maharashtra**, with native support for:

- **English**
- **Marathi (Devanagari)**
- **Marathi Romanized / Roman Marathi** (e.g., *khodkid niyantran upay*)
- **Code-Mixed Language** (Marathi with English technical terminology)

The system combines **Hybrid Retrieval-Augmented Generation (RAG)** with **Supervised Fine-Tuning (SFT)** to provide farmer-friendly, actionable responses while keeping all guidance strictly grounded in trusted agricultural resources.

---

## 1. Project Overview

Farmers frequently need fast, understandable answers regarding crop cultivation, irrigation schedules, fertilizer doses, pest control, disease management, recommended varieties, planting geometry, harvesting, and ratoon management.

The goal of this project is to build an AI-based system where a farmer can ask a question in natural language and receive an accurate, grounded answer based on authoritative agricultural research documents.

### Example

**Farmer question:**
> *ऊसातील खोडकीड कशी नियंत्रित करावी?*

**System processing flow:**
1. Understands the farmer's language and Romanized transliteration.
2. Detects the agricultural concept (`stalk_borer` / खोडकीड) and farmer intent (`pest_control`).
3. Performs dual retrieval: BGE-M3 semantic search via FAISS + corpus-wide concept scanning across all 922 knowledge chunks.
4. Ranks candidates using intent-aware hybrid scoring (semantic relevance + concept overlap + intent alignment).
5. Generates a grounded, practical answer using Gemini without hallucinating unverified chemical dosages.

---

## 2. Main Objectives

- **Grounded Agricultural Advisory**: Prevent hallucination by anchoring answers in verified sugarcane documents.
- **Multilingual & Romanized Understanding**: Native support for Devanagari Marathi, Roman Marathi, and English.
- **Distinction of Confusable Agricultural Concepts**: Strict disambiguation between distinct pests (e.g. stalk borer vs shoot borer vs top borer).
- **Dual RAG & SFT Pipeline**:
  - **RAG**: Real-time vector retrieval + LLM synthesis for direct advisory.
  - **SFT**: Curated high-quality Q&A dataset generation (balanced 1:1:1 across languages) for fine-tuning open-source LLMs (e.g., Gemma).
- **Modular & Extensible Architecture**: Reusable pipeline easily expandable to other crops (cotton, soybean, onion) and regions.

---

## 3. Scope & Knowledge Base

- **Crop**: Sugarcane (*Saccharum officinarum*)
- **Geographic Focus**: Maharashtra, India
- **Knowledge Sources**: **18 authoritative agricultural publications and research manuals** covering:
  - Sugarcane cultivation & package of practices
  - Varieties & seed cane (sett) treatment
  - Planting geometry, trench method, & spacing
  - Soil fertility, organic farming, & tillage
  - Nutrient management & fertilizer split applications (NPK, micronutrients)
  - Irrigation requirements & drip irrigation / fertigation
  - Weed control & herbicides (pre- and post-emergence)
  - Pest management (borers, white grub, aphids, pyrilla)
  - Disease management (red rot, smut, rust, wilt, grassy shoot)
  - Ratoon management (stubble shaving, gap filling, trash mulching)
  - Harvesting, maturity testing, & yield economics

---

## 4. System Architecture

### Hybrid RAG Architecture
```text
                    AGRICULTURAL PDF DATA
                              |
                              v
                     PDF TEXT EXTRACTION
                              |
                              v
                   CONSERVATIVE CLEANING
                              |
                              v
                   STRUCTURED EXTRACTION
                              |
                              v
                      SEMANTIC CHUNKING
                              |
                              v
                      CHUNK VALIDATION
                              |
                              v
                       FINAL CHUNKS
                        (922 Chunks)
                              |
                              v
                    BGE-M3 DENSE EMBEDDINGS
                              |
                              v
                     FAISS VECTOR INDEX
                     (Inner Product IP)
                              |
                              v
                    HYBRID RAG RETRIEVAL
                              |
              +---------------+---------------+
              |                               |
              v                               v
        Semantic Search                Concept Scan
         (FAISS top-k)             (Corpus Terminology)
              |                               |
              +---------------+---------------+
                              |
                              v
                    Intent-Aware Ranking
                              |
                              v
                    Relevant Source Chunks
                              |
                              v
                      GEMINI GENERATION
                              |
                              v
                   FARMER-FRIENDLY ADVISORY
```

### SFT Dataset Pipeline
```text
                 VALIDATED KNOWLEDGE CHUNKS
                           |
                           v
                     CLASSIFICATION
               (RAG Reference vs SFT Candidate)
                           |
                           v
                CANDIDATE QUALITY FILTER
                           |
                           v
                    GEMINI SFT RUNNER
                (1:1:1 Multilingual Triplet)
                           |
                           v
                     5,088 RAW Q&A
                           |
                           v
                  HEURISTIC & RULE CLEANING
                           |
                           v
                    5,059 CLEAN Q&A
                           |
                           v
                  CHUNK-LEVEL STRATIFIED SPLIT
                           |
            +--------------+--------------+
            |              |              |
            v              v              v
          TRAIN        VALIDATION        TEST
         (4,023)         (535)          (501)
                           |
                           v
                  GEMMA QLoRA TRAINING
```

---

## 5. Project Structure

```text
farmer-support-system/
├── .env.example                     # Environment template for API keys
├── .gitignore                       # Git ignore rules for caches, venvs, and data
├── README.md                        # Project documentation
├── requirements.txt                 # Pinned project dependencies
│
├── data/
│   ├── raw/
│   │   └── pdfs/                    # Original 18 agricultural PDFs
│   └── interim/
│       ├── extracted/               # Raw extracted text per page
│       ├── conservative/            # Conservative cleaned page text
│       ├── cleaned/                 # Standard cleaned pages
│       ├── structured/              # Section-aware structured text
│       ├── chunks/                  # Sentence-aligned chunks with overlap
│       ├── validated/               # Validated chunks & final 922 deduplicated chunks
│       ├── classified/              # RAG vs SFT candidate classification
│       ├── embeddings/              # Normalized BGE-M3 vector embeddings
│       ├── faiss/                   # Sugarcane FAISS vector index & metadata
│       └── sft/                     # Generated, cleaned, and split SFT datasets
│
├── src/
│   ├── __init__.py
│   ├── ingestion/                   # Raw document extraction
│   │   ├── __init__.py
│   │   └── pdf_extractor.py
│   │
│   ├── preprocessing/               # Text processing & chunking pipeline
│   │   ├── __init__.py
│   │   ├── conservative_cleaning.py
│   │   ├── clean_text.py
│   │   ├── section_detection.py
│   │   ├── chunking.py
│   │   ├── validate_chunks.py
│   │   ├── filter_validated_chunks.py
│   │   ├── finalize_chunks.py
│   │   ├── classify_chunks.py
│   │   └── filter_sft_candidates.py
│   │
│   ├── embedding/                   # Vector representations
│   │   ├── __init__.py
│   │   └── embed_chunks.py
│   │
│   ├── retrieval/                   # Vector indexing & similarity search
│   │   ├── __init__.py
│   │   ├── build_faiss_index.py
│   │   └── search_faiss.py
│   │
│   ├── rag/                         # End-to-end Hybrid RAG Advisory system
│   │   ├── __init__.py
│   │   └── rag_pipeline.py
│   │
│   └── sft/                         # Fine-tuning dataset generation & preparation
│       ├── __init__.py
│       ├── generate_farmer_sft.py
│       ├── clean_sft.py
│       └── split_sft_dataset.py
│
└── scripts/
    └── one-time/                    # One-time migration & maintenance utilities
        └── fix_embedding_ids.py
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### 1. Clone the repository
```bash
git clone https://github.com/Omkarphalle378/-FAREMR-.git
cd -FAREMR-
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Add your Gemini API key inside `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## 7. Pipeline Execution Guide

Every script can be executed directly or as a module from the project root.

### Phase 1: Data Ingestion & Extraction
Extract text from raw agricultural PDFs:
```bash
python -m src.ingestion.pdf_extractor
```

### Phase 2: Preprocessing & Semantic Chunking
Run the sequential text cleaning, structure detection, and chunking steps:
```bash
# 1. Conservative cleaning
python -m src.preprocessing.conservative_cleaning

# 2. Section and heading detection
python -m src.preprocessing.section_detection

# 3. Sentence-aligned chunking with overlap
python -m src.preprocessing.chunking

# 4. Chunk validation and artifact checks
python -m src.preprocessing.validate_chunks

# 5. Final deduplication and cleaning (produces 922 final chunks)
python -m src.preprocessing.finalize_chunks
```

### Phase 3: Classification & Candidate Filtering
Separate general RAG reference material from actionable SFT candidates:
```bash
# Classify into RAG and SFT sets
python -m src.preprocessing.classify_chunks

# Filter candidates with agricultural indicator scoring
python -m src.preprocessing.filter_sft_candidates
```

### Phase 4: Dense Embeddings & FAISS Index
Generate BGE-M3 embeddings and build the vector database:
```bash
# Generate normalized 1024-dim BGE-M3 embeddings
python -m src.embedding.embed_chunks

# Build FAISS IndexFlatIP vector database
python -m src.retrieval.build_faiss_index

# (Optional) Test similarity retrieval via CLI
python -m src.retrieval.search_faiss
```

### Phase 5: Run Hybrid RAG Advisory CLI
Launch the interactive multilingual advisory system:
```bash
python -m src.rag.rag_pipeline
```
**Try asking questions in:**
- **Marathi**: `ऊस पिकातील खोडकिडीचा बंदोबस्त कसा करावा?`
- **Roman Marathi**: `oosala kiti pani lagte?`
- **English**: `What is the recommended fertilizer schedule for sugarcane?`
- **Code-mixed**: `Sugarcane madhe drip irrigation che fayde sanga.`

### Phase 6: SFT Dataset Generation & Processing (Optional)
Generate and format instruction datasets for LLM fine-tuning:
```bash
# Generate balanced 1:1:1 multilingual Q&A pairs
python -m src.sft.generate_farmer_sft

# Clean and normalize generated Q&A
python -m src.sft.clean_sft

# Split dataset at chunk level (80% train / 10% val / 10% test)
python -m src.sft.split_sft_dataset
```

---

## 8. Technology Stack

| Component | Technology | Details |
| :--- | :--- | :--- |
| **Embeddings** | `BAAI/bge-m3` | 1024-dimensional dense multilingual embeddings |
| **Vector Index** | `faiss-cpu` | `IndexFlatIP` (Inner Product / Cosine Similarity) |
| **Generation** | Google Gemini | Grounded synthesis with fallback pool (`gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-3.5-flash-lite`) |
| **PDF Extraction** | PyMuPDF (`fitz`) | High-speed text and layout font-metric extraction |
| **SFT Fine-Tuning** | Gemma QLoRA | Prepared chunk-level stratified training data |