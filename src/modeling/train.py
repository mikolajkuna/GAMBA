# src/modeling/train.py

from pygam import LinearGAM, s, f, te
from sklearn.utils import compute_sample_weight
from src.config import INTERACTIONS, CLASS_WEIGHT_MODE

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


def train_gamba(X, y):
    weights = compute_sample_weight(
        class_weight="balanced",
        y=X[:, 1]
    )

    model = build_gamba()
    model.fit(X, y, weights=weights)

    return model, weights
