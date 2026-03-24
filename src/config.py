# src/config.py
from pathlib import Path

# =====================================================
# PATHS
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "salary_data_2009.csv"  
PROCESSED_DATA = DATA_DIR / "processed" / "salary_processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =====================================================
# REPRODUCIBILITY
# =====================================================
RANDOM_SEED = 42

# =====================================================
# FEATURES
# =====================================================
FEATURES = [
    "age",
    "gender",
    "education_level",
    "job_level",
    "experience_years",
    "distance_from_home",
    "absence",
    "child",
]
TARGET = "income"

# =====================================================
# DOMAIN RULES / CLEANING
# =====================================================
MIN_WAGE_PLN = 1276
DISTANCE_THRESHOLD_KM = 15
GENDER_MAP = {
    "M": 1, "Male": 1, "m": 1,
    "F": 0, "Female": 0, "f": 0
}
JOB_MAP = {
    1: "Junior",
    2: "Mid",
    3: "Senior",
    4: "Manager"
}
EDUCATION_MAP = {
    "High School": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4
}
JOB_LEVEL_MAP = {
    "Junior": 1,
    "Mid": 2,
    "Senior": 3,
    "Manager": 4
}

# =====================================================
# GAMBA MODEL SPEC
# =====================================================
# s() = smooth term for continuous predictors
# f() = factor/categorical term
# consistent with equation (2) in the paper
GAM_TERMS = {
    "age":               dict(type="s", constraint="monotonic_inc"),
    "education_level":   dict(type="f"),   # categorical ordinal — f() per eq. (2)
    "job_level":         dict(type="f"),   # categorical ordinal — f() per eq. (2)
    "experience_years":  dict(type="s", constraint="monotonic_inc"),
    "distance_from_home": dict(type="f"), # binary categorical — f() per eq. (2)
    "absence":           dict(type="s", constraint="monotonic_dec"),
    "child":             dict(type="s"),
}

# Interactions — both gender×job_level and gender×child per eq. (2)
INTERACTIONS = [
    {"features": ("gender", "job_level"), "lam": 10},
    {"features": ("gender", "child"),     "lam": 10},
]

# =====================================================
# MCMC PARAMETERS (Bayesian estimation)
# =====================================================
MCMC_CHAINS = 2
MCMC_DRAWS = 2000
MCMC_TUNE = 1000

# =====================================================
# TRAINING
# =====================================================
CLASS_WEIGHT_COLUMN = "gender"
CLASS_WEIGHT_MODE = "balanced"

# =====================================================
# EVALUATION
# =====================================================
MIN_GROUP_SIZE = 5
CV_EPS = 1e-6
