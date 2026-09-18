"""Vector similarity search against FAISS sugarcane knowledge base."""

import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "data" / "interim" / "faiss" / "sugarcane_faiss.index"
METADATA_FILE = PROJECT_ROOT / "data" / "interim" / "faiss" / "sugarcane_metadata.json"

MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5

_model = None
_index = None
_metadata = None


def get_resources():
    """Lazily load and cache embedding model, FAISS index, and metadata."""
    global _model, _index, _metadata

    if _model is None:
        print("Loading BGE-M3...")
        _model = SentenceTransformer(MODEL_NAME)
        print("BGE-M3 loaded successfully.")

    if _index is None:
        print("\nLoading FAISS index...")
        _index = faiss.read_index(str(INDEX_FILE))
        print(f"FAISS vectors: {_index.ntotal}")

    if _metadata is None:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        print(f"Metadata records: {len(_metadata)}")

        if _index.ntotal != len(_metadata):
            raise ValueError("FAISS index and metadata count do not match.")

    return _model, _index, _metadata


def search(query: str, top_k: int = TOP_K):
    """Search FAISS index for the top-k most similar sugarcane context chunks."""
    model, index, metadata = get_resources()

    # Convert farmer question into normalized embedding
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")

    # Search FAISS
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        result = metadata[idx].copy()
        result["score"] = float(score)
        results.append(result)

    return results


def main():
    query = input("\nEnter farmer question: ").strip()

    if not query:
        print("Please enter a question.")
        return

    results = search(query)

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("-" * 60)
        print(f"Score    : {result['score']:.4f}")
        print(f"Source   : {result['source']}")
        print(f"Page     : {result['page']}")
        print(f"Section  : {result['section']}")
        print(f"Chunk ID : {result['id']}")
        print(f"Text     : {result['text']}")


if __name__ == "__main__":
    main()
