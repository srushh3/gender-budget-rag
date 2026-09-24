from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "outputs" / "chunks.jsonl"
INDEX_FILE = PROJECT_DIR / "outputs" / "faiss.index"
METADATA_FILE = PROJECT_DIR / "outputs" / "chunk_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    chunks = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    print(f"Creating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    # Inner product works as cosine similarity because
    # the embeddings are normalized.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings.astype("float32"))

    return index


def save_metadata(chunks):
    metadata = []

    for chunk in chunks:
        metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk.get("source"),
                "document_type": chunk.get("document_type"),
                "page": chunk.get("page"),
                "row": chunk.get("row"),
                "sheet": chunk.get("sheet"),
                "chunk_index": chunk.get("chunk_index"),
                "total_chunks_from_record": chunk.get(
                    "total_chunks_from_record"
                ),
                "text": chunk.get("text"),
            }
        )

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():

    print("=" * 80)
    print("BUILDING VECTOR INDEX")
    print("=" * 80)

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks)}")

    embeddings = create_embeddings(chunks, model)

    index = create_faiss_index(embeddings)

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    save_metadata(chunks)

    print("\n" + "=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)

    print(f"Vectors stored: {index.ntotal}")
    print(f"Vector dimension: {embeddings.shape[1]}")

    print(f"\nFAISS index:")
    print(INDEX_FILE)

    print(f"\nMetadata:")
    print(METADATA_FILE)


if __name__ == "__main__":
    main()