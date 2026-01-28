import matplotlib.pyplot as plt
import numpy as np
from src.config import JOB_MAP, FIGURES_DIR

def gap_by_job_level(X, gap, job_map=JOB_MAP):
    levels = sorted(np.unique(X[:, 3]))
    mean_gaps = [gap[X[:, 3] == lvl].mean() for lvl in levels]

    plt.figure(figsize=(7, 5))
    plt.bar([job_map[int(l)] for l in levels], mean_gaps)
    plt.axhline(0, color="red", linestyle="--")
    plt.ylabel("Mean Gender Pay Gap (PLN)")
    plt.title("Gender Pay Gap by Job Level")
    plt.show()

def child_income_plot(X, preds_f, preds_m):
    child_grid = np.arange(int(X[:,7].min()), int(X[:,7].max())+1)

    plt.figure(figsize=(7,5))
    plt.plot(child_grid, preds_f, marker='o', label='Female', color='red')
    plt.plot(child_grid, preds_m, marker='o', label='Male', color='blue')
    plt.xlabel("Number of children")
    plt.ylabel("Predicted Income (PLN)")
    plt.title("Effect of Number of Children on Income by Gender")
    plt.legend()
    plt.grid(True)
    plt.show()
