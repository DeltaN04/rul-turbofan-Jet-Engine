#!/usr/bin/env python3
"""Compute Wilson 95% CIs for per-class precision/recall from existing classification CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FDS = ["FD001", "FD002", "FD003", "FD004"]
CLASS3_LABELS = ["safe", "warning", "critical"]
BIN_LABELS = ["negative", "positive"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_cis_from_classification_csv(fd: str):
    """Read master_classification.csv for a subset and compute Wilson CIs."""
    df = pd.read_csv(RES / "master_classification.csv")
    df_fd = df[df.fd == fd].copy()
    
    # Filter to relevant models: logreg, rf, hgb, hgb-via-RUL
    models_keep = ["logreg", "rf", "hgb", "hgb-via-RUL"]
    df_fd = df_fd[df_fd.model.isin(models_keep)]
    
    rows = []
    
    for _, row in df_fd.iterrows():
        task = row["task"]
        approach = row["approach"]
        model = row["model"]
        cls = row["class"]
        prec = row["precision"]
        rec = row["recall"]
        support = row["support"]
        
        if support == 0 or pd.isna(support):
            continue
        
        support = int(support)
        
        # Recall CI: TP / support, where TP = recall * support
        tp_recall = rec * support
        tp_recall_int = int(round(tp_recall))
        rec_lo, rec_hi = wilson_ci(tp_recall_int, support)
        
        # Precision CI: TP / (TP + FP), where TP = rec * support, TP+FP = TP / prec
        if prec > 0:
            tp_prec = tp_recall_int
            pred_support = int(round(tp_prec / prec)) if prec > 0 else 0
            if pred_support > 0:
                prec_lo, prec_hi = wilson_ci(tp_prec, pred_support)
            else:
                prec_lo, prec_hi = (0.0, 0.0)
        else:
            prec_lo, prec_hi = (0.0, 0.0)
        
        rows.append({
            "fd": fd,
            "task": task,
            "approach": approach,
            "model": model,
            "class": cls,
            "metric": "precision",
            "value": float(prec),
            "ci_lo": prec_lo,
            "ci_hi": prec_hi,
            "support_pred": pred_support if prec > 0 else 0
        })
        rows.append({
            "fd": fd,
            "task": task,
            "approach": approach,
            "model": model,
            "class": cls,
            "metric": "recall",
            "value": float(rec),
            "ci_lo": rec_lo,
            "ci_hi": rec_hi,
            "support_true": support
        })
    
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for fd in FDS:
        print(f"Processing {fd}...")
        df_cis = compute_cis_from_classification_csv(fd)
        all_rows.append(df_cis)
    
    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(RES / "test_classification_cis_last_cycle.csv", index=False)
    print(f"\nWrote results/test_classification_cis_last_cycle.csv ({len(df_all)} rows)")
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()