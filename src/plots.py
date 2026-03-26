# src/plots.py
import matplotlib.pyplot as plt
import numpy as np
from src.config import JOB_MAP, FIGURES_DIR, FEATURES


def gap_by_job_level(X, gap, job_map=JOB_MAP, save=True):
    """Bar chart of mean gender pay gap by job level."""
    job_idx = FEATURES.index("job_level")
    levels = sorted(np.unique(X[:, job_idx]))
    mean_gaps = [gap[X[:, job_idx] == lvl].mean() for lvl in levels]

    plt.figure(figsize=(7, 5))
    plt.bar([job_map[int(l)] for l in levels], mean_gaps, color="skyblue")
    plt.axhline(0, color="red", linestyle="--")
    plt.ylabel("Mean Gender Pay Gap (PLN)")
    plt.title("Gender Pay Gap by Job Level")
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "gap_by_job_level.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")

    plt.show()


def child_income_plot(X, preds_f, preds_m, save=True):
    """Line plot of predicted income by number of children, by gender."""
    child_idx = FEATURES.index("child")
    child_grid = np.arange(int(X[:, child_idx].min()), int(X[:, child_idx].max()) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(child_grid, preds_f, marker="o", label="Female", color="red")
    plt.plot(child_grid, preds_m, marker="o", label="Male", color="blue")
    plt.xlabel("Number of children")
    plt.ylabel("Predicted Income (PLN)")
    plt.title("Effect of Number of Children on Income by Gender")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "child_income_plot.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")

    plt.show()


def motherhood_penalty_plot(gap_by_children, hdi_low=None, hdi_high=None, save=True):
    """
    Figure 2 from paper: pay gap escalation by number of children
    with optional 94% credible interval shading (Bayesian model).
    """
    children = sorted(gap_by_children.keys())
    gaps = [gap_by_children[c] for c in children]

    plt.figure(figsize=(7, 5))
    plt.plot(children, gaps, marker="o", color="black", label="Mean gap")

    if hdi_low is not None and hdi_high is not None:
        plt.fill_between(children, hdi_low, hdi_high,
                         alpha=0.3, color="gray", label="94% credible interval")

    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Number of Children")
    plt.ylabel("Pay Gap (PLN)")
    plt.title("Motherhood Penalty: Pay Gap by Number of Children")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "motherhood_penalty.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")

    plt.show()
