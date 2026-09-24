import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "outputs" / "chunks.jsonl"


def tokenize(text):
    """Convert text into simple lowercase tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    """Load all chunks and their metadata."""
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


class BM25Retriever:
    """BM25 retriever built once and reused for multiple queries."""

    def __init__(self):
        self.chunks = load_chunks()

        tokenized_documents = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        print(f"BM25 index loaded: {len(self.chunks)} chunks")

    def search(self, query, top_k=10):
        """Search the corpus using BM25."""
        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            result = self.chunks[index].copy()
            result["bm25_score"] = float(scores[index])
            result["chunk_index"] = int(index)

            results.append(result)

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()

    while True:
        query = input("\nEnter your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = retriever.search(query, top_k=10)

        print("\nBM25 Results:\n")

        for i, result in enumerate(results, start=1):
            print(f"{i}. Score: {result['bm25_score']:.4f}")
            print(f"   Source: {result.get('source')}")
            print(f"   Page: {result.get('page')}")
            print(f"   Row: {result.get('row')}")
            print(f"   Text: {result['text'][:500]}")
            print("-" * 80)