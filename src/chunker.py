from pathlib import Path
import json
import re


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "outputs" / "raw_documents.jsonl"
OUTPUT_FILE = PROJECT_DIR / "outputs" / "chunks.jsonl"


# Chunking settings
MAX_CHARS = 1800
OVERLAP = 250


def load_records():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def clean_text(text):
    """
    Basic text cleanup.

    We intentionally do not try to 'correct' Hindi or OCR-like
    extraction artifacts because those are part of the source corpus.
    """

    text = text.replace("\x00", " ")

    # Normalize repeated whitespace while preserving the text itself.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def split_long_text(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """
    Split a long record into overlapping chunks.

    We first try to split at paragraph/newline boundaries.
    If a section is still too large, it is split by character count.
    """

    if len(text) <= max_chars:
        return [text]

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # If adding this paragraph keeps the chunk within the limit.
        if len(current) + len(paragraph) + 1 <= max_chars:
            current = (
                paragraph
                if not current
                else current + "\n" + paragraph
            )
            continue

        # Save the current chunk.
        if current:
            chunks.append(current)

        # If this individual paragraph is too large,
        # split it directly.
        if len(paragraph) > max_chars:

            start = 0

            while start < len(paragraph):

                end = start + max_chars

                piece = paragraph[start:end]

                chunks.append(piece)

                if end >= len(paragraph):
                    break

                start = end - overlap

            current = ""

        else:
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def create_chunks(records):
    chunks = []

    chunk_id = 0

    for record in records:

        text = clean_text(record.get("text", ""))

        if not text:
            continue

        text_chunks = split_long_text(text)

        for chunk_index, chunk_text in enumerate(text_chunks):

            chunk = {
                "chunk_id": f"chunk_{chunk_id:05d}",
                "text": chunk_text,

                # Original document information
                "source": record.get("source"),
                "document_type": record.get("document_type"),

                # Location information
                "page": record.get("page"),
                "row": record.get("row"),
                "sheet": record.get("sheet"),

                # Useful for debugging/retrieval
                "chunk_index": chunk_index,
                "total_chunks_from_record": len(text_chunks),
            }

            chunks.append(chunk)

            chunk_id += 1

    return chunks


def save_chunks(chunks):

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )


def main():

    print("=" * 80)
    print("CHUNKING DOCUMENTS")
    print("=" * 80)

    records = load_records()

    print(f"Input records: {len(records)}")

    chunks = create_chunks(records)

    save_chunks(chunks)

    print(f"Output chunks: {len(chunks)}")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\nSample chunks:")

    for chunk in chunks[:5]:

        print("\n" + "-" * 80)

        print("Chunk ID:", chunk["chunk_id"])
        print("Source:", chunk["source"])
        print("Page:", chunk["page"])
        print("Row:", chunk["row"])
        print("Sheet:", chunk["sheet"])
        print("Text:")
        print(chunk["text"][:500])


if __name__ == "__main__":
    main()