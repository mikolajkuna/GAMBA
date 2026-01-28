# src/modeling/__init__.py
from .train import train_gam_reml, train_gam_bayes
from .predict import predict_gam_bayes

__all__ = ["train_gam_reml", "train_gam_bayes", "predict_gam_bayes"]
