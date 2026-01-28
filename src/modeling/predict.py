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

def predict_gam_bayes(trace, X_new):
    beta0 = trace.posterior["beta0"].mean().values
    beta = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    preds = beta0 + X_new @ beta
    return preds
