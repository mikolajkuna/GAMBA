# src/plots.py

import matplotlib.pyplot as plt
import numpy as np


def gap_by_job_level(X, gap, job_map):
    levels = sorted(np.unique(X[:, 3]))
    mean_gaps = [gap[X[:, 3] == lvl].mean() for lvl in levels]

    plt.figure(figsize=(7, 5))
    plt.bar([job_map[int(l)] for l in levels], mean_gaps)
    plt.axhline(0, color="red", linestyle="--")
    plt.ylabel("Mean Gender Pay Gap (PLN)")
    plt.title("Gender Pay Gap by Job Level")
    plt.show()
