# src/dataset.py
import pandas as pd


def load_raw_salary(path):
    with open(path, "r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    return pd.read_csv(path, sep=sep)
