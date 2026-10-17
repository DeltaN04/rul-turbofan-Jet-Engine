"""Plotting helpers: consistent style, colour-blind-safe, >=200dpi."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = "colorblind"
DPI = 200


def setup():
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": DPI, "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 11})


def save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
