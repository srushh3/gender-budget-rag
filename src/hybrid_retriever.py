import json
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import sys
from query_processor import process_query
sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------
# Paths and models
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INDEX_FILE = PROJECT_DIR / "outputs" / "faiss.index"
METADATA_FILE = PROJECT_DIR / "outputs" / "chunk_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize(text):
    """Convert text into simple lowercase tokens."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------

class HybridRetriever:

    def __init__(self):

        # -------------------------------------------------
        # Load embedding model
        # -------------------------------------------------

        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # -------------------------------------------------
        # Load FAISS index
        # -------------------------------------------------

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        # -------------------------------------------------
        # Load chunk metadata
        # -------------------------------------------------

        print("Loading chunk metadata...")

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(f)

        # -------------------------------------------------
        # Build BM25 index
        # -------------------------------------------------

        print("Building BM25 index...")

        tokenized_documents = [
            tokenize(chunk["text"])
            for chunk in self.metadata
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        # -------------------------------------------------
        # Load Cross-Encoder reranker
        # -------------------------------------------------

        print("Loading reranker model...")

        self.reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

        print(
            f"\nLoaded {self.index.ntotal} FAISS vectors "
            f"and {len(self.metadata)} chunks."
        )


    # -----------------------------------------------------
    # FAISS search
    # -----------------------------------------------------

    def faiss_search(
        self,
        query,
        top_k=30
    ):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            chunk = self.metadata[index].copy()

            chunk["chunk_index"] = int(index)
            chunk["faiss_score"] = float(score)
            chunk["retrieval_method"] = "faiss"

            results.append(chunk)

        return results


    # -----------------------------------------------------
    # BM25 search
    # -----------------------------------------------------

    def bm25_search(
        self,
        query,
        top_k=30
    ):

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = scores.argsort()[::-1][
            :min(top_k, len(scores))
        ]

        results = []

        for index in ranked_indices:

            chunk = self.metadata[index].copy()

            chunk["chunk_index"] = int(index)
            chunk["bm25_score"] = float(scores[index])
            chunk["retrieval_method"] = "bm25"

            results.append(chunk)

        return results


    # -----------------------------------------------------
    # Combine FAISS + BM25 candidates
    # -----------------------------------------------------

    def retrieve_candidates(
        self,
        query,
        faiss_k=30,
        bm25_k=30
    ):

        faiss_results = self.faiss_search(
            query,
            faiss_k
        )

        bm25_results = self.bm25_search(
            query,
            bm25_k
        )

        combined = {}

        # Add FAISS results
        for result in faiss_results:

            chunk_id = result["chunk_id"]

            combined[chunk_id] = result


        # Add BM25 results
        for result in bm25_results:

            chunk_id = result["chunk_id"]

            if chunk_id in combined:

                combined[chunk_id][
                    "retrieval_method"
                ] = "faiss+bm25"

                combined[chunk_id][
                    "bm25_score"
                ] = result["bm25_score"]

            else:

                combined[chunk_id] = result


        return list(combined.values())


    # -----------------------------------------------------
    # Cross-Encoder reranking
    # -----------------------------------------------------

    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):

        if not candidates:
            return []

        # Create query-document pairs
        pairs = []

        for candidate in candidates:
            document = f"""
        Source: {candidate['source']}
        Page: {candidate.get('page')}
        Content:
        {candidate['text']}
        """
            pairs.append((query, document))

        # Calculate relevance scores
        scores = self.reranker.predict(
            pairs
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores
        ):

            result = candidate.copy()

            result["reranker_score"] = float(
                score
            )

            reranked.append(result)

        # Highest relevance first
        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return reranked[:top_k]
    
    def search(self, query, candidate_k=50, top_k=5):
        # Detect language and translate non-English queries
        query_info = process_query(query)
        retrieval_query = query_info["translated_query"]

        print(f"Original query: {query}")

        if query_info["translated"]:
            print(f"Translated query: {retrieval_query}")

        candidates = self.retrieve_candidates(
            retrieval_query,
            faiss_k=candidate_k,
            bm25_k=candidate_k
        )

        results = self.rerank(
            retrieval_query,
            candidates,
            top_k=top_k
        )

        return results

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("HYBRID RAG RETRIEVER")
    print("=" * 80)

    retriever = HybridRetriever()

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print("No question entered.")

        return


    # -----------------------------------------------------
    # Stage 1: FAISS + BM25
    # -----------------------------------------------------

    print("\nRetrieving candidates...")

    candidates = retriever.retrieve_candidates(
        query,
        faiss_k=30,
        bm25_k=30
    )

    print(
        f"Candidate pool: {len(candidates)}"
    )


    # -----------------------------------------------------
    # Stage 2: Cross-Encoder reranking
    # -----------------------------------------------------

    print("\nReranking candidates...")

    results = retriever.rerank(
        query,
        candidates,
        top_k=5
    )


    # -----------------------------------------------------
    # Display final results
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP RERANKED RESULTS")
    print("=" * 80)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"Retrieval Method: "
            f"{result.get('retrieval_method')}"
        )

        if "faiss_score" in result:

            print(
                f"FAISS Score: "
                f"{result['faiss_score']:.4f}"
            )

        if "bm25_score" in result:

            print(
                f"BM25 Score: "
                f"{result['bm25_score']:.4f}"
            )

        print(
            f"Chunk: "
            f"{result['chunk_id']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"Page: "
            f"{result.get('page')}"
        )

        print(
            f"Row: "
            f"{result.get('row')}"
        )

        print(
            f"Sheet: "
            f"{result.get('sheet')}"
        )

        print("\nText:")

        print(
            result["text"][:1000]
        )


if __name__ == "__main__":
    main()