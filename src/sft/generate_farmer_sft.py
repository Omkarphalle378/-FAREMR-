import sys
import os
import json
import time
import re
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

from dotenv import load_dotenv
from google import genai

# =========================================================
# CONFIGURATION & PATH RESOLUTION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "validated" / "final_chunks.json"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "sft"
OUTPUT_FILE = OUTPUT_FOLDER / "generated_sft.jsonl"
ENV_FILE = PROJECT_ROOT / ".env"

# =========================================================
# START / END CHUNK
# =========================================================

START_CHUNK_NUMBER = 1

# None = process until the last candidate.
END_CHUNK_NUMBER = None

# Primary model and automatic fallbacks to eliminate quota bottlenecks
MODEL_POOL = [
    "gemini-3.5-flash",       # High-capacity active quota flagship model
    "gemini-3.1-flash-lite",   # Fast active fallback model
    "gemini-3.6-flash",       # High-reasoning preview model
    "gemini-3.5-flash-lite",   # Fallback model
]

current_model_idx = 0

# Lower temperature for maximum factual consistency and zero grammar hallucination
TEMPERATURE = 0.2

MAX_OUTPUT_TOKENS = 1500

MAX_RETRIES = 6

REQUEST_DELAY = 2.5

# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv(ENV_FILE, override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "\n"
        "GEMINI_API_KEY was not found.\n\n"
        "Create a .env file in the project root containing:\n\n"
        "GEMINI_API_KEY=your_api_key_here\n"
    )

client = genai.Client(
    api_key=api_key
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)



# =========================================================
# JSON SCHEMA
# =========================================================

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    },
                    "answer": {
                        "type": "string"
                    }
                },
                "required": [
                    "question",
                    "answer"
                ]
            }
        }
    },
    "required": [
        "examples"
    ],
    "additionalProperties": False
}


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# FILTER IRRELEVANT / METADATA QUESTIONS
# =========================================================

def is_irrelevant_question(question):
    q = normalize_text(question)

    blocked_patterns = [
        r"\bwho (is|are) the author",
        r"\bwho wrote",
        r"\bauthors? of the",
        r"\bpublication date",
        r"\bprinting date",
        r"\bwhen was .* published",
        r"\bwhen was .* printed",
        r"\bwho (is|are) the publisher",
        r"\bpublisher",
        r"\btelephone number",
        r"\bphone number",
        r"\bcontact number",
        r"\bcopyright",
        r"\bisbn",
        r"\bedition",
        r"\bprinted by",
        r"\bdocument title",
        r"\btitle of the document",
        r"\bwho is thanked",
        r"\bwho were thanked",
        r"\backnowledg",
        r"\blekha(k|ak)\b",
        r"\bprakashak\b",
        r"\bsampadak\b",
        r"\bcontact kara\b"
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, q):
            return True

    return False


# =========================================================
# VALIDATE SOURCE CHUNK QUALITY
# =========================================================

def is_valid_chunk(text):
    text = str(text).strip()
    words = text.split()

    # 1. Reject very short chunks
    if len(words) < 35:
        return False, f"Too short ({len(words)} words < 35)"

    # 2. Reject incomplete sentence fragments ending on dangling connectors
    dangling_endings = {
        "that", "which", "and", "or", "with", "for", "from",
        "is", "are", "was", "were", "the", "a", "an", "of", "to", "in", "by"
    }
    last_word = re.sub(r"[^\w]", "", words[-1].lower())
    if last_word in dangling_endings:
        return False, f"Fragment ending with dangling connector '{last_word}'"

    # 3. Reject fragments starting mid-sentence
    first_word = re.sub(r"[^\w]", "", words[0].lower())
    if first_word in {"is", "are", "and", "or", "that", "which", "but"}:
        return False, f"Fragment starting with mid-sentence connector '{first_word}'"

    # 4. Reject preface / acknowledgements / administrative chunks
    admin_patterns = [
        r"\bgratitude\b", r"\bthankful to\b", r"\bprofusely thank\b",
        r"\bforeword\b", r"\bpreface\b", r"\backnowledg",
        r"\btyping the manuscript\b", r"\beditorial board\b",
        r"\bcompilation committee\b"
    ]
    for ap in admin_patterns:
        if re.search(ap, text, re.IGNORECASE):
            return False, "Administrative / Preface chunk (no agricultural content)"

    return True, "Valid"


# =========================================================
# DYNAMIC BALANCED RATIO: QUESTIONS PER LANGUAGE
# =========================================================

def get_qa_per_language_count(text):
    """
    Determine how many Q&A pairs to generate PER LANGUAGE.
    Guarantees an exact 1:1:1 balanced ratio across:
      1. English
      2. Devanagari Marathi
      3. Romanized Marathi (Manglish / WhatsApp)

    - Small chunk (< 90 words): 2 per language = 6 Q&A total
    - Medium chunk (90 - 180 words): 3 per language = 9 Q&A total
    - Large / dense chunk (> 180 words): 4 per language = 12 Q&A total
    """
    words = len(str(text).split())
    if words < 90:
        return 2
    elif words <= 180:
        return 3
    else:
        return 4


# =========================================================
# RIGOROUS QUALITY VALIDATOR FOR Q&A PAIRS
# =========================================================

def is_valid_qa_pair(question, answer, lang="all"):
    q = str(question).strip()
    a = str(answer).strip()

    q_words = q.split()
    a_words = a.split()

    # 1. Minimum sentence length (complete thoughts, not stubs)
    if len(q_words) < 4 or len(a_words) < 5:
        return False

    # 2. Question punctuation
    if not (q.endswith("?") or q.endswith("?")):
        return False

    # 3. Answer punctuation
    if not (a.endswith(".") or a.endswith("।") or a.endswith("!")):
        return False

    # 4. Answers must NOT end with dangling prepositions or connectors
    dangling = {
        "that", "the", "a", "an", "and", "or", "with", "for",
        "is", "are", "was", "were", "of", "to", "in", "by", "from",
        "as", "ki", "ani", "va"
    }
    last_word = re.sub(r"[^\w]", "", a_words[-1].lower())
    if last_word in dangling:
        return False

    # 5. Filter out meta-references / placeholder phrases
    meta_phrases = [
        "as mentioned above", "as mentioned in", "as stated in",
        "according to the text", "according to the source",
        "refer to source", "the text does not", "this operations",
        "a bunch of", "source madhye sangitle", "sandarbhanusar"
    ]
    a_lower = a.lower()
    for mp in meta_phrases:
        if mp in a_lower:
            return False

    # 6. Script purity & character validation
    if lang == "marathi_devanagari":
        # Must contain substantial Devanagari characters
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", q))
        if devanagari_chars < 5:
            return False
        devanagari_ans = len(re.findall(r"[\u0900-\u097F]", a))
        if devanagari_ans < 5:
            return False

    elif lang == "marathi_romanized":
        # Must be PURE Latin/ASCII characters (reject Gujarati, Hindi, Devanagari characters)
        try:
            q.encode("ascii")
            a.encode("ascii")
        except UnicodeEncodeError:
            return False

    elif lang == "english":
        # Must be PURE Latin/ASCII characters
        try:
            q.encode("ascii")
            a.encode("ascii")
        except UnicodeEncodeError:
            return False

    # 7. Irrelevant question check
    if is_irrelevant_question(q):
        return False

    return True


# =========================================================
# REMOVE DUPLICATE EXAMPLES
# =========================================================

def remove_duplicates(examples, lang="all"):
    unique = []
    seen = set()

    for example in examples:
        if not isinstance(example, dict):
            continue

        question = str(example.get("question", "")).strip()
        answer = str(example.get("answer", "")).strip()

        if not is_valid_qa_pair(question, answer, lang):
            continue

        key = normalize_text(question)
        if key in seen:
            continue

        seen.add(key)
        unique.append({
            "question": question,
            "answer": answer
        })

    return unique


# =========================================================
# READ ALREADY PROCESSED CHUNKS
# =========================================================

def get_processed_chunks():
    processed = set()

    if not os.path.exists(OUTPUT_FILE):
        print("\nNo existing output file found.")
        return processed

    print("\nExisting output found. Reading processed chunk IDs...")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                chunk_id = record.get("metadata", {}).get("chunk_id")
                if chunk_id:
                    processed.add(str(chunk_id))
            except json.JSONDecodeError:
                continue

    print(f"Already processed chunks: {len(processed)}")
    return processed


# =========================================================
# PERMANENT ERROR CHECK
# =========================================================

def is_permanent_error(error):
    error_text = str(error).lower()

    # Rate limit 429 / burst quota is TEMPORARY, never permanent
    if (
        "429" in error_text
        or "too_many_requests" in error_text
        or "rate limit" in error_text
        or "retry in" in error_text
    ):
        return False

    permanent_patterns = [
        "api key", "invalid api key", "authentication", "unauthenticated",
        "permission denied", "permission_denied",
        "not found", "not_found", "model not found",
        "model is not available", "invalid argument", "invalid_argument"
    ]
    for pattern in permanent_patterns:
        if pattern in error_text:
            return True
    return False


# =========================================================
# GENERATE SINGLE LANGUAGE BATCH
# =========================================================

def generate_language_batch(chunk, lang_key, count):
    text = str(chunk.get("text", "")).strip()

    if lang_key == "english":
        lang_instructions = f"""
LANGUAGE & STYLE: PURE ENGLISH (Farmer-Facing Practical Agronomy)

Generate EXACTLY {count} distinct question-answer pairs in clear, grammatically correct English.
- Focus on practical agricultural operations: recommended fertilizer doses, planting methods, irrigation scheduling, weed management, harvesting indicators, and variety traits.
- Questions should reflect real queries a sugarcane farmer, extension officer, or student would ask.
- Answers must be complete, informative, grammatically flawless English sentences.
- Strictly grounded in the SOURCE TEXT. Do not hallucinate outside information.

Format example:
- Question: "What is the recommended fertilizer dose per hectare for sugarcane?"
  Answer: "The recommended dose is 250 kg nitrogen, 115 kg phosphorus, and 115 kg potassium per hectare."
"""

    elif lang_key == "marathi_devanagari":
        lang_instructions = f"""
LANGUAGE & STYLE: AUTHENTIC DEVANAGARI MARATHI (मराठीत प्रश्न आणि उत्तर)

Generate EXACTLY {count} distinct question-answer pairs in pure, grammatically accurate Marathi (देवनागरी लिपी).

MANDATORY ACCURACY RULES:
1. GRAMMAR & SPELLING: Zero spelling mistakes, standard Marathi grammar (व्याकरणदृष्ट्या अचूक व उच्च दर्जाचे).
2. AGRICULTURAL LEXICON (कृषी संज्ञा अचूक वापरा):
   - Fertilizer -> खत / खताची मात्रा / डोस
   - Weed control -> तण नियंत्रण / तण काढणे
   - Irrigation -> पाणी व्यवस्थापन / ठिबक सिंचन (drip irrigation)
   - Planting / Sowing -> उसाची लागवड
   - Sett / Seed treatment -> बेणे प्रक्रिया
   - Harvesting -> उसाची तोडणी / काढणी
   - Maturity -> पक्वता / पक्वतेची लक्षणे
   - Yield -> हेक्टरी उत्पादन / उतारा
   - Disease / Pest -> रोग व कीड नियंत्रण
   - Intercropping -> आंतरपीक पद्धती
3. Natural Context: You may include standard English technical terms in brackets where helpful (e.g., 'ठिबक सिंचन (drip irrigation)', 'बेसल डोस (basal dose)').
4. SCRIPT PURITY: BOTH Question AND Answer MUST be written in pure Marathi Devanagari script.
5. COMPLETE ANSWERS: Every answer must be a complete, grammatically sound sentence ending with a full stop ('.').

Format example:
- प्रश्न: "उसाला खताचा पहिला बेसल डोस (basal dose) केव्हा द्यावा लागतो?"
  उत्तर: "उसाला लागवडीच्या वेळीच नत्र, स्फुरद आणि पालाशची पहिली बेसल मात्रा देणे आवश्यक असते."
- प्रश्न: "अडसाली उसाचे हेक्टरी सरासरी उत्पादन किती मिळते?"
  उत्तर: "अडसाली उसाचे हेक्टरी सरासरी उत्पादन सुमारे १५० ते १७० टन इतके मिळते."
"""

    elif lang_key == "marathi_romanized":
        lang_instructions = f"""
LANGUAGE & STYLE: ROMANIZED CODE-MIXED MARATHI (WhatsApp / Manglish Style)

Generate EXACTLY {count} distinct question-answer pairs in Romanized Marathi (Marathi written using the English alphabet).
This represents how Indian and Maharashtrian farmers naturally type on WhatsApp or mobile keyboards.

MANDATORY ACCURACY RULES:
1. STANDARD PHONETIC SPELLING (Maharashtra Standard):
   - Question words: 'kadhi' (when), 'kasa/kashi/kase' (how), 'kiti' (how much), 'konta/konti' (which), 'kay' (what).
   - Verbs: 'dyave / dila pahije' (should give), 'karave' (should do), 'takaave' (should apply), 'vaprave' (should use), 'ahe / aste' (is).
   - Farming nouns: 'usala / usamadhe' (to/in sugarcane), 'pikaat' (in crop), 'sheti' (farming), 'panyache' (of water).
2. CODE-MIXING: Natural farming loanwords in English are welcomed and expected ('drip irrigation', 'weed control', 'fertilizer dose', 'basal dose', 'yield', 'variety', 'germination').
3. BOTH Question AND Answer MUST be written in Romanized Marathi (English letters only, no Devanagari script).
4. COMPLETE ANSWERS: Every answer must be a complete, natural, helpful sentence.

Format example:
- Question: "Usala khatacha pahila basal dose kadhi dila pahije?"
  Answer: "Usala lagwadichya velich khatacha pahila basal dose dila pahije."
- Question: "Sugarcane madhe drip irrigation vaparlyane kay fayda hoto?"
  Answer: "Drip irrigation mule panyachi bachat hote ani usacha yield sudha vadhto."
- Question: "Adsali usachi lagwad kontya mahinyat karavi?"
  Answer: "Adsali usachi lagwad sadharanpane July te August darmiyan keli jate."
"""

    prompt = f"""
You are creating high-quality supervised fine-tuning (SFT) data
for a farmer-facing agricultural assistant specializing in sugarcane.

{lang_instructions}

=========================================================
SOURCE TEXT
=========================================================
{text}

=========================================================
STRICT SOURCE-GROUNDING RULES
=========================================================
1. Ground every answer STRICTLY in the SOURCE TEXT.
2. Do NOT invent doses, varieties, dates, or chemical names.
3. Preserve numbers, units, ranges, and timing exactly as in the source.
4. Do NOT create questions about authors, publishers, page numbers, or book metadata.
5. Every answer MUST be a complete, standalone sentence without dangling words.

=========================================================
OUTPUT STRUCTURE
=========================================================
Return EXACTLY this JSON structure:
{{
    "examples": [
        {{
            "question": "...",
            "answer": "..."
        }}
    ]
}}
Return JSON only. No markdown fences.
"""

    global current_model_idx

    for attempt in range(1, MAX_RETRIES + 1):
        active_model = MODEL_POOL[current_model_idx % len(MODEL_POOL)]
        try:
            interaction = client.interactions.create(
                model=active_model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": QA_SCHEMA
                }
            )

            content = interaction.output_text
            if not content:
                raise RuntimeError("Model returned an empty response.")

            clean_content = content.strip()
            if clean_content.startswith("```"):
                clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content, flags=re.IGNORECASE)
                clean_content = re.sub(r"\s*```$", "", clean_content)

            result = json.loads(clean_content)
            examples = result.get("examples", [])
            if not isinstance(examples, list):
                raise ValueError("Invalid response: 'examples' is not a list.")

            valid_examples = remove_duplicates(examples, lang_key)
            return valid_examples

        except Exception as e:
            error_str = str(e).lower()
            print(f"        Attempt {attempt}/{MAX_RETRIES} on [{active_model}] failed: {type(e).__name__}: {e}")

            # If 429 RateLimit or Quota Exceeded, switch immediately to next model in pool!
            if (
                "429" in error_str
                or "too_many_requests" in error_str
                or "quota" in error_str
                or "resource_exhausted" in error_str
            ):
                current_model_idx += 1
                next_model = MODEL_POOL[current_model_idx % len(MODEL_POOL)]
                print(f"        [Quota/RateLimit on {active_model}] Automatically failing over to '{next_model}'...")
                time.sleep(2.0)
                continue

            # If model not available or 404, rotate to next model
            if (
                "not found" in error_str
                or "not_found" in error_str
                or "no longer available" in error_str
            ):
                current_model_idx += 1
                next_model = MODEL_POOL[current_model_idx % len(MODEL_POOL)]
                print(f"        [{active_model} unavailable] Switching to '{next_model}'...")
                time.sleep(1.0)
                continue

            if is_permanent_error(e):
                raise

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 3)

    return []


# =========================================================
# GENERATE ALL BALANCED Q&A FOR ONE CHUNK
# =========================================================

def generate_chunk_qa(chunk):
    text = str(chunk.get("text", "")).strip()
    per_lang_count = get_qa_per_language_count(text)

    languages = [
        ("english", "Pure English"),
        ("marathi_devanagari", "Devanagari Marathi"),
        ("marathi_romanized", "Romanized Marathi (Manglish)")
    ]

    all_chunk_examples = []
    used_questions = set()

    for lang_key, lang_label in languages:
        print(f"      Generating {lang_label} ({per_lang_count} pairs)...")

        examples = generate_language_batch(chunk, lang_key, per_lang_count)

        added = 0
        for ex in examples:
            q = ex["question"]
            a = ex["answer"]

            if not is_valid_qa_pair(q, a, lang_key):
                continue

            norm_q = normalize_text(q)
            if norm_q in used_questions:
                continue
            used_questions.add(norm_q)

            all_chunk_examples.append({
                "question": q,
                "answer": a,
                "language": lang_key
            })
            added += 1

        print(f"        Accepted: {added}/{per_lang_count}")
        time.sleep(REQUEST_DELAY)

    return all_chunk_examples


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    print("\nLoading filtered SFT candidate chunks...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"\nInput file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    if not isinstance(all_chunks, list):
        raise ValueError("Input file must contain a JSON list.")

    total_chunks = len(all_chunks)
    print(f"Total chunks loaded: {total_chunks}")

    start_index = START_CHUNK_NUMBER - 1
    end_index = min(END_CHUNK_NUMBER or total_chunks, total_chunks)
    chunks = all_chunks[start_index:end_index]

    processed_chunks = get_processed_chunks()

    print("\n" + "=" * 70)
    print("HIGH-QUALITY BALANCED MULTILINGUAL SFT GENERATION")
    print("=" * 70)
    print(f"Total candidates       : {total_chunks}")
    print(f"Starting item number   : {START_CHUNK_NUMBER}")
    print(f"Ending item number     : {end_index}")
    print(f"Chunks selected        : {len(chunks)}")
    print(f"Ratio per chunk        : Exact 1:1:1 (English : Marathi Devanagari : Romanized)")
    print(f"Primary Model          : {MODEL_POOL[0]}")
    print(f"Fallback Models        : {', '.join(MODEL_POOL[1:])}")
    print(f"Output                 : {OUTPUT_FILE}")
    print(f"Previously processed   : {len(processed_chunks)}")
    print("=" * 70 + "\n")

    total_new_examples = 0
    successful_chunks = 0
    skipped_chunks = 0

    try:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as output:
            for position, chunk in enumerate(chunks, start=START_CHUNK_NUMBER):
                chunk_id = str(chunk.get("id", f"chunk_{position}"))

                if chunk_id in processed_chunks:
                    skipped_chunks += 1
                    continue

                print("-" * 70)
                print(f"[{position}/{total_chunks}] Chunk: {chunk_id} | Topic: {chunk.get('topic', '')}")
                print(f"Source: {chunk.get('source', '')} | Page: {chunk.get('page', '')}")
                print("-" * 70)

                text = str(chunk.get("text", "")).strip()
                is_valid, reason = is_valid_chunk(text)
                if not is_valid:
                    print(f"    SKIPPED ({reason})")
                    skipped_chunks += 1
                    continue

                examples = generate_chunk_qa(chunk)

                if examples:
                    for ex in examples:
                        record = {
                            "question": ex["question"],
                            "answer": ex["answer"],
                            "language": ex["language"],
                            "metadata": {
                                "source": chunk.get("source", ""),
                                "page": chunk.get("page", ""),
                                "section": chunk.get("section", ""),
                                "chunk_id": chunk_id,
                                "topic": chunk.get("topic", ""),
                                "language_type": ex["language"]
                            }
                        }
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_new_examples += 1

                    output.flush()
                    processed_chunks.add(chunk_id)
                    successful_chunks += 1
                    print(f"    --> Total saved for chunk: {len(examples)} Q&A pairs (Balanced 1:1:1)")
                else:
                    print(f"    --> Warning: 0 valid Q&A pairs generated.")

                print()

    except KeyboardInterrupt:
        print("\n\nGeneration paused by user. All saved Q&A pairs remain intact.")

    print("\n" + "=" * 70)
    print("GENERATION RUN SUMMARY")
    print("=" * 70)
    print(f"Successful chunks : {successful_chunks}")
    print(f"Skipped chunks    : {skipped_chunks}")
    print(f"Total new Q&A     : {total_new_examples}")
    print(f"Output location   : {OUTPUT_FILE}")
    print("=" * 70 + "\n")
