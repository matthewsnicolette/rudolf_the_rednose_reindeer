"""
Generic CSV evaluation script.
Converted and generalised from the original notebook so it can be reused for
any CSV file, regardless of the number of rows or columns.
"""

from pathlib import Path
import sys
import pandas as pd


def load_data(csv_file: str) -> pd.DataFrame:
    """Load any CSV file using a few common encodings."""
    encodings = ["utf-8", "latin1", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(csv_file, encoding=enc)
        except Exception as exc:
            last_error = exc

    raise last_error


def evaluate_csv(df: pd.DataFrame) -> None:
    """Print a generic summary for any tabular CSV dataset."""
    print("\n=== DATA PREVIEW ===")
    print(df.head())

    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== COLUMN NAMES ===")
    print(list(df.columns))

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATE ROWS ===")
    print(df.duplicated().sum())

    print("\n=== DESCRIBE (ALL COLUMNS) ===")
    print(df.describe(include="all").transpose())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print("\n=== CORRELATION MATRIX (NUMERIC COLUMNS ONLY) ===")
        print(df[numeric_cols].corr(numeric_only=True))
    else:
        print("\n=== CORRELATION MATRIX ===")
        print("No numeric columns available.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python Python_Workshop_converted.py <path_to_csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = load_data(str(csv_path))
    evaluate_csv(df)


if __name__ == "__main__":
    main()
