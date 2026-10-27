#!/usr/bin/env python3
"""Compute per-class precision/recall CIs for last-cycle test evaluation (FAST version)."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
SEED = 42
np.random.seed(SEED)

FDS = ["FD001", "FD002", "FD003", "FD004"]
CLASS3_LABELS = ["safe", "warning", "critical"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_test_last_cycle(fd: str):
    """Load test data and return last-cycle true labels."""
    from src.data import load_split, add_rul_test_all_cycles, add_class_labels
    train, test, rul_last = load_split("data/raw", fd)
    test_all = add_rul_test_all_cycles(test, rul_last)
    test_all = add_class_labels(test_all, col="RUL")
    last_idx = test_all.groupby("unit")["cycle"].idxmax().sort_values()
    test_last = test_all.loc[last_idx].copy()
    return test_last


def main():
    rows = []
    
    for fd in FDS:
        print(f"Processing {fd}...")
        test_last = load_test_last_cycle(fd)
        y3_true = test_last["cls3_id"].values
        yb_true = test_last["bin30"].values
        units = test_last["unit"].values
        
        # Load summary to get best model
        summ = json.load(open(RES / f"{fd}_summary.json"))
        best_model = summ["best_model"]
        baselines = ["logreg", "rf", "hgb"]
        models_to_eval = [best_model] + [m for m in baselines if m != best_model]
        
        # Load feature store for predictions
        from src.train import build_feature_store, prepare_subset
        from src.models import get_classifiers, get_regressors
        from src.train import rul_to_classes
        
        train, test_all, _ = prepare_subset(fd, "data/raw", CFG)
        store = build_feature_store(train.copy(), test_all.copy(), CFG, fd)
        Xtr, Xte = store["Xtr"], store["Xte"]
        
        last_idx = test_all.groupby("unit")["cycle"].idxmax().sort_values()
        pos = {ix: i for i, ix in enumerate(test_all.index)}
        last_pos = np.array([pos[i] for i in last_idx.values])
        
        for model_name in models_to_eval:
            print(f"  {model_name}...")
            # Load classifiers (fast mode: 100 iter)
            clf_name = "hgb" if model_name == "hgb_tuned" else model_name
            clf3 = get_classifiers(seed=SEED, fast=True)[clf_name].fit(Xtr, train["cls3_id"].values)
            clfb = get_classifiers(seed=SEED, fast=True)[clf_name].fit(Xtr, train["bin30"].values)
            
            # Direct predictions
            p3 = clf3.predict(Xte[last_pos])
            pb = clfb.predict(Xte[last_pos])
            
            # Regress-then-threshold predictions
            if model_name == "hgb_tuned":
                best_params = summ.get("best_hgb_params", {})
                from sklearn.ensemble import HistGradientBoostingRegressor
                reg = HistGradientBoostingRegressor(**best_params, random_state=SEED).fit(Xtr, train["RUL_cap"].values)
            else:
                reg = get_regressors(seed=SEED, fast=True)[model_name if model_name in ("ridge", "rf", "hgb", "mlp") else summ["best_model"]].fit(Xtr, train["RUL_cap"].values)
            r3, rb = rul_to_classes(reg.predict(Xte[last_pos]))
            
            # For each approach
            for approach, p3_pred, pb_pred in [
                ("direct", p3, pb),
                ("regress-threshold", r3, rb)
            ]:
                model_label = f"{model_name}-via-RUL" if approach == "regress-threshold" else model_name
                
                # 3-class metrics
                for i, cname in enumerate(CLASS3_LABELS):
                    # Precision
                    mask_pred = (p3_pred == i)
                    if mask_pred.sum() > 0:
                        tp = ((p3_pred == i) & (y3_true == i)).sum()
                        prec = tp / mask_pred.sum()
                        lo, hi = wilson_ci(int(tp), int(mask_pred.sum()))
                        rows.append({"fd": fd, "model": model_label, "approach": approach,
                                     "task": "3class", "class": cname, "metric": "precision",
                                     "value": float(prec), "ci_lo": lo, "ci_hi": hi, "support_pred": int(mask_pred.sum())})
                    # Recall
                    mask_true = (y3_true == i)
                    if mask_true.sum() > 0:
                        tp = ((p3_pred == i) & (y3_true == i)).sum()
                        rec = tp / mask_true.sum()
                        lo, hi = wilson_ci(int(tp), int(mask_true.sum()))
                        rows.append({"fd": fd, "model": model_label, "approach": approach,
                                     "task": "3class", "class": cname, "metric": "recall",
                                     "value": float(rec), "ci_lo": lo, "ci_hi": hi, "support_true": int(mask_true.sum())})
                
                # Binary metrics
                for cls_name, cls_val in [("negative", 0), ("positive", 1)]:
                    # Precision
                    mask_pred = (pb_pred == cls_val)
                    if mask_pred.sum() > 0:
                        tp = ((pb_pred == cls_val) & (yb_true == cls_val)).sum()
                        prec = tp / mask_pred.sum()
                        lo, hi = wilson_ci(int(tp), int(mask_pred.sum()))
                        rows.append({"fd": fd, "model": model_label, "approach": approach,
                                     "task": "binary30", "class": cls_name, "metric": "precision",
                                     "value": float(prec), "ci_lo": lo, "ci_hi": hi, "support_pred": int(mask_pred.sum())})
                    # Recall
                    mask_true = (yb_true == cls_val)
                    if mask_true.sum() > 0:
                        tp = ((pb_pred == cls_val) & (yb_true == cls_val)).sum()
                        rec = tp / mask_true.sum()
                        lo, hi = wilson_ci(int(tp), int(mask_true.sum()))
                        rows.append({"fd": fd, "model": model_label, "approach": approach,
                                     "task": "binary30", "class": cls_name, "metric": "recall",
                                     "value": float(rec), "ci_lo": lo, "ci_hi": hi, "support_true": int(mask_true.sum())})
    
    df = pd.DataFrame(rows)
    df.to_csv(RES / "test_classification_cis_last_cycle.csv", index=False)
    print(f"\nWrote results/test_classification_cis_last_cycle.csv ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()