# src/features.py

import numpy as np
import pandas as pd

from src.config import (
    FEATURES,
    TARGET,
    GENDER_MAP,
    MIN_WAGE_PLN,
    DISTANCE_THRESHOLD_KM,
)


# =====================================================
# === PREPROCESSING ==================================
# =====================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw salary dataset and apply domain rules.
    No side effects. No I/O.
    """

    df = df.copy()

    # --- gender encoding ---
    df["gender"] = df["gender"].map(GENDER_MAP)

    # --- numeric coercion ---
    numeric_cols = FEATURES + [TARGET]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- distance binarization ---
    df["distance_from_home"] = (
        df["distance_from_home"] >= DISTANCE_THRESHOLD_KM
    ).astype(int)

    # --- domain filters ---
    df = df[
        (df[TARGET] >= MIN_WAGE_PLN) &
        (df[numeric_cols].ge(0).all(axis=1))
    ]

    return df.dropna()


# =====================================================
# === FEATURE MATRIX =================================
# =====================================================
def make_xy(df: pd.DataFrame):
    """
    Create X, y matrices for modeling.
    """
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y


# =====================================================
# === OPTIONAL: SAVE PROCESSED =======================
# =====================================================
def preprocess_and_save(df: pd.DataFrame, path):
    """
    Convenience helper for pipelines (optional).
    """
    processed = preprocess(df)
    processed.to_csv(path, index=False)
    return processed
