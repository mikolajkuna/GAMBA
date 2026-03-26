# src/dataset.py
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA


def load_raw_salary(path: str | Path) -> pd.DataFrame:
    """
    Load raw salary CSV file.
    Separator is inferred from the first line.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    return pd.read_csv(path, sep=sep)


if __name__ == "__main__":
    df = load_raw_salary(RAW_DATA)
    print(f"Loaded {len(df)} rows from {RAW_DATA}")
    print(df.head())
    print(df.dtypes)
