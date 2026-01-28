import numpy as np
from pygam import LinearGAM, s, f, te
import pymc as pm

def train_gam_reml(X, y, weights=None):
    gam = LinearGAM(
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
    gam.fit(X, y, weights=weights)
    return gam

def train_gam_bayes(X, y, prior="weak"):
    """
    Bayesian GAM approximation using PyMC
    prior: "weak" | "informative"
    """
    n_features = X.shape[1]
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=5000)
        beta0 = pm.Normal("beta0", mu=0, sigma=10000)
        beta = pm.Normal("beta", mu=0, sigma=2000 if prior=="informative" else 10000, shape=n_features)

        mu = beta0 + pm.math.dot(X, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(draws=2000, tune=1000, chains=4, target_accept=0.9)

    return model, trace
