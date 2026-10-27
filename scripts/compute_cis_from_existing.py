#!/usr/bin/env python3
"""Compute per-class precision/recall CIs using PRE-TRAINED models (fastest)."""
from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = ROOT / "models"
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
SEED = 42
np.random.seed(SEED)

FDS = ["FD001", "FD002", "FD003", "FD004"]
CLASS3_LABELS = ["safe", "warning", "critical"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    rows = []
    
    for fd in FDS:
        print(f"Processing {fd}...")
        # Load test data
        from src.data import load_split, add_rul_test_all_cycles, add_class_labels
        train, test, rul_last = load_split("data/raw", fd)
        test_all = add_rul_test_all_cycles(test, rul_last)
        test_all = add_class_labels(test_all, col="RUL")
        
        last_idx = test_all.groupby("unit")["cycle"].idxmax().sort_values()
        test_last = test_all.loc[last_idx].copy()
        
        y3_true = test_last["cls3_id"].values
        yb_true = test_last["bin30"].values
        
        # Load pre-trained model bundle
        with open(MODELS / f"{fd}_hgb.pkl", "rb") as f:
            bundle = pickle.load(f)
        
        model = bundle["model"]
        feat_cols = bundle["feat_cols"]
        sensors = bundle["sensors"]
        norm = bundle["norm"]
        windows = bundle["windows"]
        scaler = bundle["scaler"]
        alpha = bundle.get("smooth_alpha", 0.3)
        
        # Preprocess test data for this model
        from src.preprocess import feature_matrix, smooth_per_unit
        
        Xte_list = []
        for u in sorted(test_all["unit"].unique()):
            hist = test_all[test_all.unit == u].sort_values("cycle")
            # Condition normalise
            sm_n = norm.transform(hist.copy())
            # Smooth
            sm = smooth_per_unit(sm_n, sensors, alpha)
            for c in sensors:
                sm_n[c] = sm[c + "_sm"]
            # Rolling features
            Xf, _ = feature_matrix(sm_n, sensors, windows)
            Xf = Xf[feat_cols].fillna(0.0)
            Xs = scaler.transform(Xf.values)
            Xte_list.append(Xs[-1:])  # last cycle only
        
        Xte_last = np.vstack(Xte_list)
        
        # Predict
        pred_rul = model.predict(Xte_last)
        r3, rb = rul_to_classes(pred_rul)
        
        # Direct predictions - need to load classifiers
        from src.models import get_classifiers
        clf3 = get_classifiers(seed=SEED, fast=True)["hgb"].fit(
            *[bundle["Xtr"] if "Xtr" in bundle else None][0], train["cls3_id"].values
        ) if False else None
        
        # Actually, the model bundle doesn't store classifiers. 
        # We need to train them quickly on the training features.
        # But we don't have Xtr in the bundle. Let's use a simpler approach:
        # The classification results already exist in master_classification.csv
        # We can compute CIs from the existing predictions stored there.
        pass
    
    print("Need different approach - use existing classification CSVs")

if __name__ == "__main__":
    main()