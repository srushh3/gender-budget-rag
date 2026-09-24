from pathlib import Path

import pandas as pd
from pypdf import PdfReader


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def inspect_pdf(file_path):
    print("\n" + "=" * 80)
    print(f"PDF: {file_path.name}")
    print("=" * 80)

    try:
        reader = PdfReader(str(file_path))

        print(f"Number of pages: {len(reader.pages)}")

        for page_number, page in enumerate(reader.pages[:3], start=1):
            text = page.extract_text() or ""

            print(f"\n--- Page {page_number} ---")
            print(text[:1500])

    except Exception as e:
        print(f"ERROR reading PDF: {e}")


def inspect_csv(file_path):
    print("\n" + "=" * 80)
    print(f"CSV: {file_path.name}")
    print("=" * 80)

    try:
        df = pd.read_csv(file_path)

        print(f"Shape: {df.shape}")

        print("\nColumns:")
        print(list(df.columns))

        print("\nFirst 5 rows:")
        print(df.head().to_string())

    except Exception as e:
        print(f"ERROR reading CSV: {e}")


def inspect_xls(file_path):
    print("\n" + "=" * 80)
    print(f"XLS: {file_path.name}")
    print("=" * 80)

    try:
        excel_file = pd.ExcelFile(file_path)

        print("Sheets:")
        print(excel_file.sheet_names)

        for sheet_name in excel_file.sheet_names:
            print(f"\n--- Sheet: {sheet_name} ---")

            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=None
            )

            print(f"Shape: {df.shape}")

            print("\nFirst 15 rows:")
            print(df.head(15).to_string(index=True, header=False))

    except Exception as e:
        print(f"ERROR reading XLS: {e}")


def main():
    print(f"Inspecting documents from: {DATA_DIR}")

    files = sorted(DATA_DIR.iterdir())

    for file_path in files:
        if file_path.suffix.lower() == ".pdf":
            inspect_pdf(file_path)

        elif file_path.suffix.lower() == ".csv":
            inspect_csv(file_path)

        elif file_path.suffix.lower() == ".xls":
            inspect_xls(file_path)

        else:
            print(f"Skipping unsupported file: {file_path.name}")


if __name__ == "__main__":
    main()