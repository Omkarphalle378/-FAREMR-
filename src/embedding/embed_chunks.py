"""Vector embedding generation for finalized knowledge chunks using BGE-M3."""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "validated" / "final_chunks.json"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "interim" / "embeddings"
OUTPUT_FILE = OUTPUT_FOLDER / "bge_m3_embeddings.json"
MODEL_NAME = "BAAI/bge-m3"


def generate_embeddings(
    input_path: Path = INPUT_FILE,
    output_path: Path = OUTPUT_FILE,
    model_name: str = MODEL_NAME,
    batch_size: int = 8
) -> None:
    """Generate dense normalized embeddings with BGE-M3 and save along with chunk metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"LOADING EMBEDDING MODEL: {model_name}")
    print("=" * 60)

    model = SentenceTransformer(model_name)
    print(f"{model_name} loaded successfully!")

    print("\nLoading chunks...")
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks: {len(chunks)}")

    texts = [chunk.get("text", "").strip() for chunk in chunks if chunk.get("text", "").strip()]
    print(f"Texts to embed: {len(texts)}")

    print("\nGenerating embeddings (may take some time on CPU)...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    ).tolist()

    output_data = []
    for chunk, embedding in zip(chunks, embeddings):
        output_data.append({
            "chunk_id": chunk.get("chunk_id", chunk.get("id")),
            "source": chunk.get("source"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "text": chunk.get("text"),
            "embedding": embedding
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("BGE-M3 EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Chunks embedded : {len(output_data)}")
    print(f"Embedding size  : {len(embeddings[0])}")
    print(f"Model           : {model_name}")
    print(f"\nSaved:\n{output_path}")


def main():
    generate_embeddings()


if __name__ == "__main__":
    main()
