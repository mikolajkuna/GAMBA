# src/features.py

from src.config import FEATURES, TARGET, GENDER_MAP, MIN_WAGE_PLN, DISTANCE_THRESHOLD_KM, EDUCATION_MAP, JOB_LEVEL_MAP
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# =====================================================
# === PREPROCESSING ==================================
# =====================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw salary dataset and apply domain rules.
    No side effects. No I/O.
    """

    df = df.copy()

    # --- encode categorical features ---
    df["gender"] = df["gender"].map(GENDER_MAP)
    df["education_level"] = df["education_level"].map(EDUCATION_MAP)
    df["job_level"] = df["job_level"].map(JOB_LEVEL_MAP)

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


# =====================================================
# === OBLICZANIE WAGI PRÓBEK =========================
# =====================================================
def compute_sample_weights(df: pd.DataFrame):
    """
    Compute the class weights for the gender feature (balanced).
    """
    # Użyj 'compute_class_weight' do obliczenia wag dla 'gender'
    df = df.copy()
    gender_classes = df["gender"].astype(int).values  # upewnij się, że to są liczby 0/1
    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(gender_classes),  # klasy: 0 i 1
        y=gender_classes
    )

    # Mapowanie wag do oryginalnego dataframe'u
    weight_dict = dict(zip(np.unique(gender_classes), class_weights))
    df["sample_weight"] = df["gender"].map(weight_dict)

    return df
