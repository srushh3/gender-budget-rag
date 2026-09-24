import json
from hybrid_retriever import HybridRetriever


def load_questions(path):
    questions = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def main():
    retriever = HybridRetriever()

    questions = load_questions("questions_dev.jsonl")

    for item in questions:
        question_id = item["question_id"]
        question = item["question"]

        print("\n" + "=" * 80)
        print(f"{question_id}: {question}")
        print("=" * 80)

        results = retriever.search(
            question,
            candidate_k=50,
            top_k=5
        )

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Reranker Score: {result['reranker_score']:.4f}")
            print(f"Method: {result.get('method', 'N/A')}")
            print(f"Source: {result['source']}")
            print(f"Page: {result.get('page')}")
            print(f"Row: {result.get('row')}")
            print(f"Sheet: {result.get('sheet')}")
            print(f"Text:\n{result['text'][:700]}")


if __name__ == "__main__":
    main()