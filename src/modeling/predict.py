# src/modeling/predict.py

import numpy as np


def predict(model, X):
    return model.predict(X)


def counterfactual_gender_gap(model, X):
    X_m = X.copy()
    X_f = X.copy()

    X_m[:, 1] = 1
    X_f[:, 1] = 0

    return model.predict(X_m) - model.predict(X_f)
