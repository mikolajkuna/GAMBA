import numpy as np
import pandas as pd

from src.config import (
    FEATURES,
    TARGET,
    GENDER_MAP,
    MIN_WAGE_PLN,
    DISTANCE_THRESHOLD_KM,
    EDUCATION_MAP,
    JOB_LEVEL_MAP
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
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(GENDER_MAP)

    # --- education encoding ---
    if "education_level" in df.columns:
        df["education_level"] = df["education_level"].map(EDUCATION_MAP)

    # --- job level encoding ---
    if "job_level" in df.columns:
        df["job_level"] = df["job_level"].map(JOB_LEVEL_MAP)

    # --- numeric coercion ---
    numeric_cols = FEATURES + [TARGET]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- distance binarization ---
    if "distance_from_home" in df.columns:
        df["distance_from_home"] = (
            df["distance_from_home"] >= DISTANCE_THRESHOLD_KM
        ).astype(int)

    # --- domain filters ---
    df = df[
        df[TARGET].ge(MIN_WAGE_PLN) &
        df[numeric_cols].ge(0).all(axis=1)
    ]

    return df.dropna()


# =====================================================
# === FEATURE MATRIX =================================
# =====================================================
def make_xy(df: pd.DataFrame):
    """
    Create X, y matrices for modeling.
    """
    X = df[FEATURES].astype(np.float32).to_numpy()
    y = df[TARGET].astype(np.float32).to_numpy()
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
