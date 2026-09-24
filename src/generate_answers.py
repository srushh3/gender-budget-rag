import argparse
import json
import sys
from pathlib import Path

from hybrid_retriever import HybridRetriever
from llm_generator import generate_answer


BASE_DIR = Path(__file__).resolve().parent.parent


def load_questions(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Generate grounded answers from the RAG retriever.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    args = parser.parse_args()

    question_path = BASE_DIR / args.questions if not Path(args.questions).is_absolute() else Path(args.questions)
    output_path = BASE_DIR / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.reconfigure(encoding="utf-8")
    retriever = HybridRetriever()

    with open(output_path, "w", encoding="utf-8") as out:
        for question in load_questions(question_path):
            question_id = question.get("question_id") or question.get("id")
            query = question.get("question") or question.get("text")

            if not question_id or not query:
                print(f"Skipping malformed question: {question}")
                continue

            print(f"\nGenerating {question_id}: {query}")

            try:
                results = retriever.search(
                    query,
                    candidate_k=args.candidate_k,
                    top_k=args.top_k,
                )

                generated = generate_answer(query, results)
            except Exception as exc:
                print(f"Generation failed for {question_id}: {exc}")
                generated = {
                    "answered": False,
                    "answer": "",
                    "sources": [],
                }

            record = {
                "question_id": question_id,
                "answered": generated["answered"],
                "answer": generated["answer"],
                "sources": generated["sources"],
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps(record, ensure_ascii=False))

    print(f"\nSaved answers to: {output_path}")


if __name__ == "__main__":
    main()
