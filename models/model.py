# model.py

import numpy as np
from sklearn.utils import compute_sample_weight
from pygam import LinearGAM, s, f, te


FEATURES = [
    "age", "gender", "education_level", "job_level",
    "experience_years", "distance_from_home", "absence", "child"
]


def prepare_xy(df):
    X = df[FEATURES].astype(np.float32).to_numpy()
    y = df["income"].astype(np.float32).to_numpy()
    return X, y


def compute_weights(X):
    return compute_sample_weight(
        class_weight="balanced",
        y=X[:, 1]   # gender
    )


def build_gamba():
    return LinearGAM(
        s(0, constraints="monotonic_inc") +
        f(1) +
        s(2, constraints="monotonic_inc") +
        s(3, constraints="monotonic_inc") +
        s(4, constraints="monotonic_inc") +
        f(5) +
        s(6, constraints="monotonic_dec") +
        s(7) +
        te(1, 3, lam=10)
    )
