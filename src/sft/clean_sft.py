import json
import os
import re
import tempfile
from collections import Counter


from pathlib import Path

# ============================================================
# CONFIGURATION & PATH RESOLUTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "sft" / "clean_sft.jsonl"

# We overwrite the same file after targeted cleaning.
OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "sft" / "clean_sft.jsonl"



# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    records = []
    invalid_lines = 0

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                if isinstance(record, dict):
                    records.append(record)

            except json.JSONDecodeError:

                invalid_lines += 1

    return records, invalid_lines


# ============================================================
# SAVE JSONL SAFELY
# ============================================================

def save_jsonl_safely(path, records):
    directory = os.path.dirname(os.path.abspath(str(path)))


    # Temporary file is created only during the operation.
    # It is deleted after replacement.
    fd, temp_path = tempfile.mkstemp(
        suffix=".jsonl",
        dir=directory
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8"
        ) as f:

            for record in records:

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        os.replace(
            temp_path,
            path
        )

    except Exception:

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize(text):

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GET METADATA
# ============================================================

def get_metadata(record):

    metadata = record.get(
        "metadata",
        {}
    )

    if isinstance(metadata, dict):
        return metadata

    return {}


def get_chunk_id(record):

    return get_metadata(record).get(
        "chunk_id",
        ""
    )


def get_language(record):

    language = record.get(
        "language"
    )

    if language:
        return language

    return get_metadata(record).get(
        "language_type",
        "unknown"
    )


# ============================================================
# PUBLICATION METADATA QUESTIONS
#
# These are clearly about the publication itself rather than
# sugarcane/agricultural knowledge.
# ============================================================

def is_publication_metadata_question(question):

    q = normalize(question)

    patterns = [

        # Received / accepted dates
        r"\bon what date was the paper\b.*\breceived\b",
        r"\bwhen was the paper\b.*\breceived\b",
        r"\bwhen was the paper\b.*\baccepted\b",
        r"\bwhen was the research article\b.*\baccepted\b",
        r"\bwhen was the research article\b.*\breceived\b",

        # Paper title
        r"\bwhat is the title of the research paper\b",
        r"\bwhat is the title of the paper\b",
        r"\bwhat is the title of the research article\b",

        # Journal/publication information
        r"\bwhat is the name of the journal\b",
        r"\bwhich journal published\b",
        r"\bwhich journal\b.*\bpublished\b",

        # Authors
        r"\bwho is the author of the paper\b",
        r"\bwho authored the paper\b",
        r"\bwho wrote the paper\b",

        # Contact information
        r"\bcorresponding author\b",
        r"\bwhat is the email address\b",
        r"\bwhat is the contact number\b",
        r"\bwhat is the telephone number\b",

        # Publication identifiers
        r"\bwhat is the issn\b",
        r"\bwhat is the isbn\b",
        r"\bwhat is the doi\b",
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            q
        ):

            return True

    # Marathi publication metadata
    marathi_patterns = [

        "शोधनिबंधाची स्वीकृती",
        "शोधनिबंध स्वीकारला",
        "शोधनिबंधाचे शीर्षक",
        "लेखाचे शीर्षक",
        "शोधनिबंध कधी स्वीकारला",
    ]

    for pattern in marathi_patterns:

        if pattern in question:
            return True

    return False


# ============================================================
# RESEARCH REFERENCE QUESTIONS
#
# Remove questions whose main purpose is asking what a named
# researcher/study found.
#
# IMPORTANT:
# We are NOT checking the section name.
# Therefore useful agricultural information from a
# "Literature review" or "General" section can remain.
# ============================================================

def is_reference_question(question):

    q = normalize(question)

    patterns = [

        r"\bwhat did .* et al\.? find\b",

        r"\bwhat did .* et al\.? indicate\b",

        r"\bwhat did .* et al\.? demonstrate\b",

        r"\bwhat did .* et al\.? discover\b",

        r"\bwhat did the research by\b",

        r"\bwhat did the study by\b",

        r"\bwhat did the research of\b",

        r"\bwhat did the study of\b",

        r"\bwhat did .* researchers find\b",

        r"\bwhat did .* researchers discover\b",

        r"\bwhat did .* research show\b",

        r"\bwhich researcher studied\b",

        r"\bwhich researchers studied\b",

        r"\bwho studied\b",

        r"\bwho reported\b",
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            q
        ):

            return True

    return False


# ============================================================
# ADMINISTRATIVE / DOCUMENT QUESTIONS
#
# Only remove questions that are about publishing the
# document/technical bulletin itself.
# ============================================================

def is_administrative_question(question):

    q = normalize(question)

    patterns = [

        r"\bwhich institution\b.*\bpublishing\b",
        r"\bwhich institution\b.*\bpublished\b",
        r"\bwhich organization\b.*\bpublished\b",
        r"\bwhat institution\b.*\bpublication\b",
        r"\bwhich coordination unit\b.*\bpublishing\b",
        r"\bwhich coordination unit\b.*\btechnical bulletin\b",
        r"\bwho contributed to publishing\b",
        r"\bwho helped publish\b",
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            q
        ):

            return True

    return False


# ============================================================
# NON-ANSWER
# ============================================================

def is_non_answer(record):

    answer = normalize(
        record.get(
            "answer",
            ""
        )
    )

    patterns = [

        "source material does not provide",

        "source does not provide",

        "not available in the source",

        "not available in the document",

        "information is unavailable",

        "not mentioned",

        "not specified",

        "cannot be determined",

        "does not provide",

        "doesn't provide",

        "no information",
    ]

    for pattern in patterns:

        if pattern in answer:

            return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TARGETED FINAL SFT CLEANING")
    print("=" * 70)

    if not os.path.exists(
        INPUT_FILE
    ):

        print(
            "\nERROR: File not found:"
        )

        print(
            INPUT_FILE
        )

        return

    # --------------------------------------------------------
    # Load existing clean dataset
    # --------------------------------------------------------

    records, invalid_lines = load_jsonl(
        INPUT_FILE
    )

    print(
        f"\nInput records: {len(records)}"
    )

    print(
        f"Invalid JSON lines: {invalid_lines}"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    removed = []

    reasons = Counter()

    final_records = []

    seen_qa = set()

    seen_questions = set()

    # --------------------------------------------------------
    # Process every record
    # --------------------------------------------------------

    for record in records:

        question = str(
            record.get(
                "question",
                ""
            )
        ).strip()

        answer = str(
            record.get(
                "answer",
                ""
            )
        ).strip()

        language = get_language(
            record
        )

        reason = None

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if not question:

            reason = "missing_question"

        elif not answer:

            reason = "missing_answer"

        # ----------------------------------------------------
        # Clearly bad publication question
        # ----------------------------------------------------

        elif is_publication_metadata_question(
            question
        ):

            reason = (
                "publication_metadata_question"
            )

        # ----------------------------------------------------
        # Clearly bad reference question
        # ----------------------------------------------------

        elif is_reference_question(
            question
        ):

            reason = (
                "reference_question"
            )

        # ----------------------------------------------------
        # Clearly administrative question
        # ----------------------------------------------------

        elif is_administrative_question(
            question
        ):

            reason = (
                "administrative_question"
            )

        # ----------------------------------------------------
        # Non-answer
        # ----------------------------------------------------

        elif is_non_answer(
            record
        ):

            reason = (
                "possible_non_answer"
            )

        # ----------------------------------------------------
        # Exact duplicate Q&A
        # ----------------------------------------------------

        if reason is None:

            qa_key = (

                normalize(question),

                normalize(answer)

            )

            if qa_key in seen_qa:

                reason = (
                    "duplicate_complete_qa"
                )

            else:

                seen_qa.add(
                    qa_key
                )

        # ----------------------------------------------------
        # Exact duplicate question in same language
        # ----------------------------------------------------

        if reason is None:

            question_key = (

                language,

                normalize(question)

            )

            if question_key in seen_questions:

                reason = (
                    "duplicate_question_same_language"
                )

            else:

                seen_questions.add(
                    question_key
                )

        # ----------------------------------------------------
        # Keep / remove
        # ----------------------------------------------------

        if reason:

            reasons[
                reason
            ] += 1

            removed.append({

                "question":
                    question,

                "answer":
                    answer,

                "language":
                    language,

                "chunk_id":
                    get_chunk_id(record),

                "reason":
                    reason
            })

        else:

            final_records.append(
                record
            )

    # --------------------------------------------------------
    # Save directly back to clean_sft.jsonl
    # --------------------------------------------------------

    save_jsonl_safely(
        OUTPUT_FILE,
        final_records
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    languages = Counter()

    chunks = set()

    for record in final_records:

        languages[
            get_language(record)
        ] += 1

        chunk_id = get_chunk_id(
            record
        )

        if chunk_id:
            chunks.add(
                chunk_id
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(
        f"\nInput records       : {len(records)}"
    )

    print(
        f"Final records       : {len(final_records)}"
    )

    print(
        f"Removed records     : {len(removed)}"
    )

    print(
        f"Final unique chunks : {len(chunks)}"
    )

    print("\n" + "-" * 70)
    print("REMOVAL BREAKDOWN")
    print("-" * 70)

    if reasons:

        for reason, count in sorted(
            reasons.items()
        ):

            print(
                f"{reason:40} : {count}"
            )

    else:

        print(
            "No records removed."
        )

    print("\n" + "-" * 70)
    print("FINAL LANGUAGE DISTRIBUTION")
    print("-" * 70)

    for language, count in languages.items():

        print(
            f"{language:30} : {count}"
        )

    print("\n" + "-" * 70)
    print("IMPORTANT")
    print("-" * 70)

    print(
        "\nRecords were removed only when the QUESTION itself"
    )

    print(
        "was clearly publication/reference/administrative"
    )

    print(
        "content or the ANSWER was a non-answer."
    )

    print(
        "\nNo record was removed only because of its section."
    )

    print(
        "\nExisting clean_sft.jsonl was updated."
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
