# src/dataset.py

from pathlib import Path
import pandas as pd


def load_raw_salary(path: str | Path) -> pd.DataFrame:
    """
    Load raw salary CSV file.
    Separator is inferred from the first line.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","

    return pd.read_csv(path, sep=sep)
