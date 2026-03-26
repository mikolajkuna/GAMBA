# src/features.py
from src.config import (
    FEATURES, TARGET, GENDER_MAP,
    MIN_WAGE_PLN, PROCESSED_DATA, RANDOM_SEED
)
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw salary dataset and apply domain rules.
    No side effects. No I/O.
    """
    df = df.copy()

    # encode gender (M/F → 1/0)
    df["gender"] = df["gender"].map(GENDER_MAP)

    # education_level, job_level, distance_from_home already numeric in dataset

    # numeric coercion
    numeric_cols = FEATURES + [TARGET]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # domain filters
    df = df[
        (df[TARGET] >= MIN_WAGE_PLN) &
        (df[numeric_cols].ge(0).all(axis=1))
    ]
    return df.dropna()


def make_xy(df: pd.DataFrame):
    """Create X, y matrices for modeling."""
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y


def preprocess_and_save(df: pd.DataFrame, path):
    """Convenience helper for pipelines (optional)."""
    processed = preprocess(df)
    processed.to_csv(path, index=False)
    return processed


def compute_sample_weights(df: pd.DataFrame):
    """Compute balanced class weights for gender feature."""
    gender_classes = df["gender"].astype(int).values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(gender_classes),
        y=gender_classes
    )
    weight_dict = dict(zip(np.unique(gender_classes), class_weights))
    df = df.copy()
    df["sample_weight"] = df["gender"].map(weight_dict)
    return df


if __name__ == "__main__":
    from src.dataset import load_raw_salary
    from src.config import RAW_DATA
    df_raw = load_raw_salary(RAW_DATA)
    df = preprocess(df_raw)
    preprocess_and_save(df, PROCESSED_DATA)
    print(f"Processed {len(df)} rows → {PROCESSED_DATA}")
    print(df.describe())
