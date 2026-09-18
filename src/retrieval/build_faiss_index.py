"""Build FAISS vector index from precomputed BGE-M3 embeddings."""

import json
from pathlib import Path
import faiss
import numpy as np

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "embeddings" / "bge_m3_embeddings_fixed.json"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "faiss"
INDEX_FILE = OUTPUT_FOLDER / "sugarcane_faiss.index"
METADATA_FILE = OUTPUT_FOLDER / "sugarcane_metadata.json"


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOADING BGE-M3 EMBEDDINGS")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records: {len(data)}")

    embeddings = []
    metadata = []

    for item in data:
        embedding = item.get("embedding")
        if embedding is None:
            continue

        embeddings.append(embedding)
        metadata.append({
            "id": item["id"],
            "source": item["source"],
            "page": item["page"],
            "section": item["section"],
            "text": item["text"]
        })

    # Convert embeddings to NumPy array
    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2:
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")

    number_of_vectors = embeddings.shape[0]
    embedding_dimension = embeddings.shape[1]

    print(f"Vectors          : {number_of_vectors}")
    print(f"Embedding size   : {embedding_dimension}")

    # Create FAISS index
    print("\nCreating FAISS index...")
    # Because BGE-M3 embeddings are normalized, inner product equals cosine similarity
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    print(f"FAISS vectors    : {index.ntotal}")

    if index.ntotal != len(metadata):
        raise ValueError("FAISS vector count does not match metadata count.")

    # Save FAISS index and metadata
    faiss.write_index(index, str(INDEX_FILE))

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("FAISS INDEX CREATION COMPLETE")
    print("=" * 60)
    print(f"Vectors indexed  : {index.ntotal}")
    print(f"Dimension        : {embedding_dimension}")
    print("Similarity       : Inner Product / Cosine")
    print("\nSaved files:")
    print(INDEX_FILE)
    print(METADATA_FILE)


if __name__ == "__main__":
    main()
