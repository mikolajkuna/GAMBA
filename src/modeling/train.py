# src/modeling/train.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.utils import compute_sample_weight
from pygam import LinearGAM, s, f, te
import matplotlib.pyplot as plt
from src.config import (
    GENDER_MAP, FEATURES, GAM_TERMS, INTERACTIONS, TARGET, 
    JOB_MAP, MIN_WAGE_PLN, DISTANCE_THRESHOLD_KM, 
    CLASS_WEIGHT_MODE, MIN_GROUP_SIZE
)


# =====================================================
# === LOAD & PREPROCESS DATA =========================
# =====================================================
def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    return pd.read_csv(path, sep=sep)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].map(GENDER_MAP)

    numeric_cols = FEATURES  # Wszystkie cechy

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # distance_from_home jako binarna 0/1
    df["distance_from_home"] = (df["distance_from_home"] >= DISTANCE_THRESHOLD_KM).astype(int)

    # Filtrujemy minimalną płacę i brak ujemnych wartości
    df = df[(df[TARGET] >= MIN_WAGE_PLN) & (df[numeric_cols].ge(0).all(axis=1))]
    return df.dropna()


# =====================================================
# === TRAIN GAMBA MODEL ==============================
# =====================================================
def train_gamba(synthetic_path: str):
    # Wczytanie i przetworzenie danych
    synthetic_df = preprocess(load_csv(synthetic_path))

    X = synthetic_df[FEATURES].astype(np.float32).to_numpy()
    y = synthetic_df[TARGET].astype(np.float32).to_numpy()

    # =====================================================
    # === SAMPLE WEIGHTS (BALANCE GENDER) ================
    # =====================================================
    weights = compute_sample_weight(
        class_weight=CLASS_WEIGHT_MODE,
        y=X[:, FEATURES.index("gender")]  # gender column
    )

    # =====================================================
    # === GAMBA MODEL ====================================
    # =====================================================
    terms = None  # Zaczynamy od None

    # Dodajemy terminy z configu (GAM_TERMS)
    for feature, spec in GAM_TERMS.items():
        if spec["type"] == "s":
            term = s(FEATURES.index(feature), constraints=spec.get("constraint"))
        elif spec["type"] == "f":
            term = f(FEATURES.index(feature))
        elif spec["type"] == "te":
            term = te(FEATURES.index(spec["features"][0]), FEATURES.index(spec["features"][1]), lam=spec["lam"])
        
        # Dodajemy term do istniejących terminów
        if terms is None:
            terms = term
        else:
            terms = terms + term

    # Dodajemy terminy interakcji z INTERACTIONS
    for interaction in INTERACTIONS:
        feature_idx_1 = FEATURES.index(interaction["features"][0])
        feature_idx_2 = FEATURES.index(interaction["features"][1])
        term = te(feature_idx_1, feature_idx_2, lam=interaction["lam"])
        terms = terms + term

    # Tworzymy model z wszystkimi terminami
    gamba = LinearGAM(terms)

    # Uczenie modelu
    gamba.fit(X, y, weights=weights)

    # =====================================================
    # === PERFORMANCE ====================================
    # =====================================================
    preds = gamba.predict(X)
    mae = mean_absolute_error(y, preds)
    cv = np.std(y - preds) / np.mean(y) * 100

    # =====================================================
    # === COUNTERFACTUAL GENDER PAY GAP ===================
    # =====================================================
    X_cf_m = X.copy()
    X_cf_f = X.copy()
    X_cf_m[:, FEATURES.index("gender")] = 1  # Ustawiamy płeć na M dla wszystkich
    X_cf_f[:, FEATURES.index("gender")] = 0  # Ustawiamy płeć na F dla wszystkich

    pred_m = gamba.predict(X_cf_m)
    pred_f = gamba.predict(X_cf_f)
    gap_adjusted = pred_m - pred_f

    # =====================================================
    # === PRINTING RESULTS ==============================
    # =====================================================
    print("\n--- GAMBA ---")
    print("GCV:", gamba.statistics_["GCV"])
    print("AIC:", gamba.statistics_["AIC"])
    print("\n--- PERFORMANCE (GAMBA) ---")
    print(f"MAE: {mae:,.0f} PLN")
    print(f"CV: {cv:.2f}%")

    print("\n--- GENDER PAY GAP (GAMBA) ---")
    print(f"Adjusted Mean gap (counterfactual):   {gap_adjusted.mean():,.0f} PLN")
    print(f"Adjusted Median gap (counterfactual): {np.median(gap_adjusted):,.0f} PLN")

    print("\n--- GAP BY JOB LEVEL (Adjusted / Counterfactual) ---")
    for lvl in sorted(np.unique(X[:, FEATURES.index("job_level")])):
        mask = X[:, FEATURES.index("job_level")] == lvl
        if mask.sum() < MIN_GROUP_SIZE:
            continue
        print(
            f"Job level {JOB_MAP[int(lvl)]} | "
            f"mean gap: {gap_adjusted[mask].mean():,.0f} PLN | "
            f"n={mask.sum()}"
        )

    # =====================================================
    # === BAR PLOT: GAP BY JOB LEVEL =====================
    # =====================================================
    job_levels = sorted(np.unique(X[:, FEATURES.index("job_level")]))
    mean_gaps = [gap_adjusted[X[:, FEATURES.index("job_level")] == lvl].mean() for lvl in job_levels]

    plt.figure(figsize=(7, 5))
    plt.bar([JOB_MAP[int(lvl)] for lvl in job_levels], mean_gaps, color='skyblue')
    plt.xlabel("Job Level")
    plt.ylabel("Mean Gender Pay Gap (PLN)")
    plt.title("GAMBA: Mean Gender Pay Gap by Job Level")
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

    return gamba, gap_adjusted
