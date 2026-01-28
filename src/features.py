# src/features.py

import numpy as np


FEATURES = [
    "age", "gender", "education_level", "job_level",
    "experience_years", "distance_from_home", "absence", "child"
]


def preprocess(df):
    df = df.copy()

    gender_map = {"M": 1, "Male": 1, "m": 1,
                  "F": 0, "Female": 0, "f": 0}
    df["gender"] = df["gender"].map(gender_map)

    numeric_cols = FEATURES + ["income"]

    for col in numeric_cols:
        df[col] = df[col].astype(float)

    df["distance_from_home"] = (df["distance_from_home"] >= 15).astype(int)

    df = df[
        (df["income"] >= 1276) &
        (df[numeric_cols].ge(0).all(axis=1))
    ]

    return df.dropna()


def make_xy(df):
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["income"].to_numpy(dtype=np.float32)
    return X, y
