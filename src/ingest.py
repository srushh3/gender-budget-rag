from pathlib import Path

import pandas as pd
from pypdf import PdfReader


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def extract_pdf(file_path):
    """
    Extract text from every page of a PDF.

    Returns a list of dictionaries containing:
    - text
    - source
    - page
    - document type
    """

    records = []

    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        records.append(
            {
                "text": text,
                "source": file_path.name,
                "page": page_number,
                "document_type": "pdf",
            }
        )

    return records


def extract_csv(file_path):
    """
    Extract CSV content.

    The corpus contains a CSV that does not decode as UTF-8,
    so we try a few common encodings.
    """

    encodings = [
        "utf-8",
        "cp1252",
        "latin1",
    ]

    df = None
    used_encoding = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(
            f"Could not decode CSV file: {file_path.name}"
        )

    records = []

    for index, row in df.iterrows():
        row_parts = []

        for column in df.columns:
            value = row[column]

            if pd.isna(value):
                continue

            row_parts.append(f"{column}: {value}")

        text = " | ".join(row_parts).strip()

        if not text:
            continue

        records.append(
            {
                "text": text,
                "source": file_path.name,
                "page": None,
                "row": int(index) + 2,
                "document_type": "csv",
                "encoding": used_encoding,
            }
        )

    return records


def extract_xls(file_path):
    """
    Extract content from Excel workbook.

    We intentionally read without assuming a header because
    stat20.xls contains narrative text and irregular table layout.
    """

    records = []

    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None
        )

        if df.empty:
            continue

        for row_number, row in df.iterrows():

            row_parts = []

            for value in row:

                if pd.isna(value):
                    continue

                value = str(value).strip()

                if value:
                    row_parts.append(value)

            text = " | ".join(row_parts).strip()

            if not text:
                continue

            records.append(
                {
                    "text": text,
                    "source": file_path.name,
                    "page": None,
                    "sheet": sheet_name,
                    "row": int(row_number) + 1,
                    "document_type": "xls",
                }
            )

    return records


def load_documents():
    """
    Load every supported document from the data directory.
    """

    all_records = []

    files = sorted(DATA_DIR.iterdir())

    for file_path in files:

        suffix = file_path.suffix.lower()

        print(f"Processing: {file_path.name}")

        if suffix == ".pdf":
            records = extract_pdf(file_path)

        elif suffix == ".csv":
            records = extract_csv(file_path)

        elif suffix == ".xls":
            records = extract_xls(file_path)

        else:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        print(f"  Extracted records: {len(records)}")

        all_records.extend(records)

    return all_records


def save_records(records):
    """
    Save extracted records as JSONL.

    JSONL means one JSON object per line.
    """

    import json

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "raw_documents.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(f"\nSaved extracted records to: {output_file}")


def main():

    print("=" * 80)
    print("DOCUMENT INGESTION")
    print("=" * 80)

    records = load_documents()

    print("\n" + "=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)

    print(f"Total records extracted: {len(records)}")

    save_records(records)


if __name__ == "__main__":
    main()