import json
import os
import random
from collections import defaultdict, Counter
from pathlib import Path

# ============================================================
# CONFIGURATION & PATH RESOLUTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "sft" / "clean_sft.jsonl"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "sft"

TRAIN_FILE = OUTPUT_FOLDER / "train.jsonl"
VALIDATION_FILE = OUTPUT_FOLDER / "validation.jsonl"
TEST_FILE = OUTPUT_FOLDER / "test.jsonl"
REPORT_FILE = OUTPUT_FOLDER / "split_report.json"

# Dataset split ratios
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# Reproducibility
RANDOM_SEED = 42



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

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

                if isinstance(
                    record,
                    dict
                ):

                    records.append(
                        record
                    )

            except json.JSONDecodeError:

                invalid_lines += 1

    return records, invalid_lines


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    path,
    records
):

    with open(
        path,
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


# ============================================================
# GET CHUNK ID
# ============================================================

def get_chunk_id(record):

    metadata = record.get(
        "metadata",
        {}
    )

    return metadata.get(
        "chunk_id"
    )


# ============================================================
# GET LANGUAGE
# ============================================================

def get_language(record):

    language = record.get(
        "language"
    )

    if language:
        return language

    metadata = record.get(
        "metadata",
        {}
    )

    return metadata.get(
        "language_type",
        "unknown"
    )


# ============================================================
# GROUP RECORDS BY CHUNK
# ============================================================

def group_by_chunk(records):

    chunks = defaultdict(list)

    missing_chunk_records = []

    for record in records:

        chunk_id = get_chunk_id(
            record
        )

        if not chunk_id:

            missing_chunk_records.append(
                record
            )

            continue

        chunks[
            chunk_id
        ].append(
            record
        )

    return chunks, missing_chunk_records


# ============================================================
# CHUNK INFORMATION
# ============================================================

def chunk_language_distribution(
    chunk_records
):

    languages = set()

    for record in chunk_records:

        languages.add(
            get_language(record)
        )

    return languages


# ============================================================
# CREATE STRATIFIED CHUNK SPLIT
# ============================================================

def create_split(
    chunks,
    random_seed
):

    random.seed(
        random_seed
    )

    # --------------------------------------------------------
    # We assign chunks based on their dominant language.
    #
    # This gives us better language representation across
    # train / validation / test while still keeping each
    # chunk completely together.
    # --------------------------------------------------------

    language_groups = defaultdict(list)

    for chunk_id, records in chunks.items():

        languages = Counter(
            get_language(record)
            for record in records
        )

        dominant_language = (
            languages.most_common(1)[0][0]
        )

        language_groups[
            dominant_language
        ].append(
            chunk_id
        )

    train_chunks = []
    validation_chunks = []
    test_chunks = []

    # --------------------------------------------------------
    # Split each language group independently.
    # --------------------------------------------------------

    for language, chunk_ids in language_groups.items():

        random.shuffle(
            chunk_ids
        )

        total = len(
            chunk_ids
        )

        train_count = int(
            total * TRAIN_RATIO
        )

        validation_count = int(
            total * VALIDATION_RATIO
        )

        # Make sure at least one chunk remains
        # for test when possible.
        if total >= 3:

            if train_count < 1:
                train_count = 1

            if validation_count < 1:
                validation_count = 1

            test_count = (
                total
                - train_count
                - validation_count
            )

            if test_count < 1:

                test_count = 1

                if train_count > validation_count:
                    train_count -= 1
                else:
                    validation_count -= 1

        else:

            # Very small groups
            train_count = max(
                0,
                total - 1
            )

            validation_count = 0

            test_count = (
                total
                - train_count
            )

        train_chunks.extend(
            chunk_ids[
                :train_count
            ]
        )

        validation_chunks.extend(
            chunk_ids[
                train_count:
                train_count
                + validation_count
            ]
        )

        test_chunks.extend(
            chunk_ids[
                train_count
                + validation_count:
            ]
        )

    return (
        train_chunks,
        validation_chunks,
        test_chunks
    )


# ============================================================
# CHECK CHUNK LEAKAGE
# ============================================================

def check_chunk_leakage(
    train_chunks,
    validation_chunks,
    test_chunks
):

    train_set = set(
        train_chunks
    )

    validation_set = set(
        validation_chunks
    )

    test_set = set(
        test_chunks
    )

    train_validation = (
        train_set
        & validation_set
    )

    train_test = (
        train_set
        & test_set
    )

    validation_test = (
        validation_set
        & test_set
    )

    leakage = {

        "train_validation":
            sorted(
                train_validation
            ),

        "train_test":
            sorted(
                train_test
            ),

        "validation_test":
            sorted(
                validation_test
            )
    }

    has_leakage = any(
        len(value) > 0
        for value in leakage.values()
    )

    return has_leakage, leakage


# ============================================================
# GET RECORDS FROM CHUNKS
# ============================================================

def records_from_chunks(
    chunks,
    chunk_ids
):

    records = []

    for chunk_id in chunk_ids:

        records.extend(
            chunks[
                chunk_id
            ]
        )

    return records


# ============================================================
# DATASET STATISTICS
# ============================================================

def dataset_statistics(
    records,
    chunk_ids
):

    languages = Counter()

    topics = Counter()

    sources = Counter()

    chunks_per_language = defaultdict(set)

    for record in records:

        language = get_language(
            record
        )

        languages[
            language
        ] += 1

        metadata = record.get(
            "metadata",
            {}
        )

        topics[
            metadata.get(
                "topic",
                "unknown"
            )
        ] += 1

        sources[
            metadata.get(
                "source",
                "unknown"
            )
        ] += 1

        chunk_id = get_chunk_id(
            record
        )

        chunks_per_language[
            language
        ].add(
            chunk_id
        )

    return {

        "records":
            len(records),

        "chunks":
            len(chunk_ids),

        "languages":
            dict(languages),

        "topics":
            dict(topics),

        "sources":
            dict(sources),

        "chunks_by_language":
            {
                language: len(
                    chunk_set
                )
                for language, chunk_set
                in chunks_per_language.items()
            }
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SFT TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_FILE
    ):

        print(
            "\nERROR: Input file not found:"
        )

        print(
            INPUT_FILE
        )

        return

    # --------------------------------------------------------
    # Validate ratios
    # --------------------------------------------------------

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(
        total_ratio - 1.0
    ) > 0.0001:

        print(
            "\nERROR: Split ratios must equal 1.0"
        )

        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    records, invalid_lines = load_jsonl(
        INPUT_FILE
    )

    print(
        f"\nInput records: {len(records)}"
    )

    print(
        f"Invalid JSON lines: "
        f"{invalid_lines}"
    )

    # --------------------------------------------------------
    # Group by chunk
    # --------------------------------------------------------

    chunks, missing_chunk_records = group_by_chunk(
        records
    )

    print(
        f"Unique chunks: {len(chunks)}"
    )

    print(
        f"Records without chunk_id: "
        f"{len(missing_chunk_records)}"
    )

    # --------------------------------------------------------
    # Stop if metadata is broken
    # --------------------------------------------------------

    if missing_chunk_records:

        print(
            "\nERROR:"
        )

        print(
            "Some records do not contain metadata.chunk_id."
        )

        print(
            "The dataset must be fixed before splitting."
        )

        return

    if len(chunks) < 3:

        print(
            "\nERROR: Not enough chunks for a "
            "train/validation/test split."
        )

        return

    # --------------------------------------------------------
    # Create chunk split
    # --------------------------------------------------------

    print(
        "\nCreating chunk-level split..."
    )

    (
        train_chunk_ids,
        validation_chunk_ids,
        test_chunk_ids
    ) = create_split(
        chunks,
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    has_leakage, leakage = check_chunk_leakage(
        train_chunk_ids,
        validation_chunk_ids,
        test_chunk_ids
    )

    if has_leakage:

        print(
            "\nERROR: CHUNK LEAKAGE DETECTED!"
        )

        print(
            json.dumps(
                leakage,
                indent=2,
                ensure_ascii=False
            )
        )

        print(
            "\nFiles were NOT created."
        )

        return

    print(
        "\nChunk leakage check: PASSED"
    )

    # --------------------------------------------------------
    # Convert chunks to records
    # --------------------------------------------------------

    train_records = records_from_chunks(
        chunks,
        train_chunk_ids
    )

    validation_records = records_from_chunks(
        chunks,
        validation_chunk_ids
    )

    test_records = records_from_chunks(
        chunks,
        test_chunk_ids
    )

    # --------------------------------------------------------
    # Shuffle records inside each split
    #
    # This does NOT cause leakage because chunks have already
    # been assigned exclusively to one split.
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        train_records
    )

    random.shuffle(
        validation_records
    )

    random.shuffle(
        test_records
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    save_jsonl(
        TRAIN_FILE,
        train_records
    )

    save_jsonl(
        VALIDATION_FILE,
        validation_records
    )

    save_jsonl(
        TEST_FILE,
        test_records
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    train_stats = dataset_statistics(
        train_records,
        train_chunk_ids
    )

    validation_stats = dataset_statistics(
        validation_records,
        validation_chunk_ids
    )

    test_stats = dataset_statistics(
        test_records,
        test_chunk_ids
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {

        "input_file":
            INPUT_FILE,

        "input_records":
            len(records),

        "input_chunks":
            len(chunks),

        "split_ratios": {

            "train":
                TRAIN_RATIO,

            "validation":
                VALIDATION_RATIO,

            "test":
                TEST_RATIO
                },

        "random_seed":
            RANDOM_SEED,

        "chunk_leakage":
            False,

        "train":
            train_stats,

        "validation":
            validation_stats,

        "test":
            test_stats
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal records : {len(records)}"
    )

    print(
        f"Total chunks  : {len(chunks)}"
    )

    print("\n" + "-" * 70)
    print("TRAIN")
    print("-" * 70)

    print(
        f"Chunks  : {len(train_chunk_ids)}"
    )

    print(
        f"Records : {len(train_records)}"
    )

    print("\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)

    print(
        f"Chunks  : {len(validation_chunk_ids)}"
    )

    print(
        f"Records : {len(validation_records)}"
    )

    print("\n" + "-" * 70)
    print("TEST")
    print("-" * 70)

    print(
        f"Chunks  : {len(test_chunk_ids)}"
    )

    print(
        f"Records : {len(test_records)}"
    )

    print("\n" + "-" * 70)
    print("LANGUAGE DISTRIBUTION")
    print("-" * 70)

    print(
        "\nTRAIN:"
    )

    for language, count in (
        train_stats["languages"].items()
    ):

        print(
            f"  {language:25} : {count}"
        )

    print(
        "\nVALIDATION:"
    )

    for language, count in (
        validation_stats["languages"].items()
    ):

        print(
            f"  {language:25} : {count}"
        )

    print(
        "\nTEST:"
    )

    for language, count in (
        test_stats["languages"].items()
    ):

        print(
            f"  {language:25} : {count}"
        )

    print("\n" + "-" * 70)
    print("OUTPUT FILES")
    print("-" * 70)

    print(
        f"\nTrain:"
    )

    print(
        TRAIN_FILE
    )

    print(
        f"\nValidation:"
    )

    print(
        VALIDATION_FILE
    )

    print(
        f"\nTest:"
    )

    print(
        TEST_FILE
    )

    print(
        f"\nReport:"
    )

    print(
        REPORT_FILE
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
