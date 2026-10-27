#!/usr/bin/env python3
"""Compute last-cycle test-set class counts and per-class precision/recall CIs.

Uses OFFICIAL evaluation: last cycle per test engine, labels from RUL_FD00x.txt.
Outputs:
- results/test_class_counts_last_cycle.csv
- results/test_classification_cis_last_cycle.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import (SENSORS, add_class_labels, add_piecewise_rul, add_rul_test_all_cycles,
                      add_rul_train, load_split)
from src.preprocess import (ConditionNormalizer, feature_matrix, make_sequences,
                            select_sensors_by_variance, smooth_per_unit)
from src.train import (build_feature_store, grouped_cv_classification, oof_predict,
                       prepare_subset, rul_to_classes)
from src.models import get_classifiers, get_regressors
from src.evaluate import bootstrap_ci_by_engine

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
SEED = 42
np.random.seed(SEED)

FDS = ["FD001", "FD002", "FD003", "FD004"]
CLASS3_LABELS = ["safe", "warning", "critical"]
BIN_LABELS = ["negative", "positive"]  # RUL > 30, RUL <= 30


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# Cache for feature stores and data
_data_cache = {}
_model_cache = {}

def get_cached_data(fd: str):
    """Get cached train, test_all, store, last_pos for a subset."""
    if fd not in _data_cache:
        train, test_all, _ = prepare_subset(fd, "data/raw", CFG)
        store = build_feature_store(train.copy(), test_all.copy(), CFG, fd)
        Xtr, Xte = store["Xtr"], store["Xte"]
        groups = train["unit"].values

        last_idx = test_all.groupby("unit")["cycle"].idxmax().sort_values()
        pos = {ix: i for i, ix in enumerate(test_all.index)}
        last_pos = np.array([pos[i] for i in last_idx.values])

        y3_last = test_all["cls3_id"].values[last_pos]
        yb_last = test_all["bin30"].values[last_pos]
        units_last = test_all["unit"].values[last_pos]

        _data_cache[fd] = {
            "train": train, "test_all": test_all, "store": store,
            "Xtr": Xtr, "Xte": Xte, "groups": groups,
            "last_pos": last_pos,
            "y3_true": y3_last, "yb_true": yb_last, "units": units_last
        }
    return _data_cache[fd]


def get_predictions_for_model(fd: str, model_name: str):
    """Get last-cycle predictions for a given model on a subset."""
    data = get_cached_data(fd)
    Xtr, Xte = data["Xtr"], data["Xte"]
    train = data["train"]
    last_pos = data["last_pos"]
    y3_last = data["y3_true"]
    yb_last = data["yb_true"]
    units_last = data["units"]

    # Get regressor (for regress-then-threshold) - use FAST mode (100 iters) for speed
    regressor_names = ("ridge", "rf", "hgb", "mlp", "hgb_tuned")
    cache_key = (fd, model_name, "reg")
    if cache_key not in _model_cache:
        if model_name == "hgb_tuned":
            summ = json.load(open(RES / f"{fd}_summary.json"))
            best_params = summ.get("best_hgb_params", {})
            from sklearn.ensemble import HistGradientBoostingRegressor
            reg = HistGradientBoostingRegressor(**best_params, random_state=SEED).fit(Xtr, train["RUL_cap"].values)
        elif model_name in regressor_names:
            reg = get_regressors(seed=SEED, fast=True)[model_name].fit(Xtr, train["RUL_cap"].values)
        else:
            # logreg is classifier-only; use deployed regressor for RTT
            summ = json.load(open(RES / f"{fd}_summary.json"))
            deployed = summ["best_model"]
            if deployed == "hgb_tuned":
                best_params = summ.get("best_hgb_params", {})
                from sklearn.ensemble import HistGradientBoostingRegressor
                reg = HistGradientBoostingRegressor(**best_params, random_state=SEED).fit(Xtr, train["RUL_cap"].values)
            else:
                reg = get_regressors(seed=SEED, fast=True)[deployed].fit(Xtr, train["RUL_cap"].values)
        _model_cache[cache_key] = reg
    else:
        reg = _model_cache[cache_key]

    # Get classifiers - use FAST mode
    cache_key3 = (fd, model_name, "clf3")
    cache_keyb = (fd, model_name, "clfb")
    if model_name in ("ridge", "rf", "hgb", "mlp", "hgb_tuned", "logreg"):
        clf_name = "hgb" if model_name == "hgb_tuned" else model_name
        if cache_key3 not in _model_cache:
            clf3 = get_classifiers(seed=SEED, fast=True)[clf_name].fit(Xtr, train["cls3_id"].values)
            _model_cache[cache_key3] = clf3
        else:
            clf3 = _model_cache[cache_key3]
        if cache_keyb not in _model_cache:
            clfb = get_classifiers(seed=SEED, fast=True)[clf_name].fit(Xtr, train["bin30"].values)
            _model_cache[cache_keyb] = clfb
        else:
            clfb = _model_cache[cache_keyb]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Predictions
    p3 = clf3.predict(Xte[last_pos])
    pb = clfb.predict(Xte[last_pos])
    try:
        pb_score = clfb.predict_proba(Xte[last_pos])[:, 1]
    except Exception:
        pb_score = pb.astype(float)
    # Regress-then-threshold
    r3, rb = rul_to_classes(reg.predict(Xte[last_pos]))

    return {
        "y3_true": y3_last, "y3_pred_direct": p3, "y3_pred_rtt": r3,
        "yb_true": yb_last, "yb_pred_direct": pb, "yb_pred_rtt": rb, "yb_score": pb_score,
        "units": units_last
    }


def compute_last_cycle_counts():
    """Compute safe/warning/critical counts at last cycle per test engine."""
    rows = []
    for fd in FDS:
        train, test_all, rul_last = prepare_subset(fd, "data/raw", CFG)
        last_idx = test_all.groupby("unit")["cycle"].idxmax().sort_values()
        test_last = test_all.loc[last_idx].copy()

        n_safe = int((test_last["cls3"] == "safe").sum())
        n_warn = int((test_last["cls3"] == "warning").sum())
        n_crit = int((test_last["cls3"] == "critical").sum())
        n_pos = int((test_last["bin30"] == 1).sum())
        n_neg = int((test_last["bin30"] == 0).sum())
        rows.append({
            "fd": fd,
            "n_engines": int(test_last["unit"].nunique()),
            "safe": n_safe,
            "warning": n_warn,
            "critical": n_crit,
            "positive_bin30": n_pos,
            "negative_bin30": n_neg,
            "note": "counts at LAST cycle per engine (official evaluation)"
        })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "test_class_counts_last_cycle.csv", index=False)
    print("Wrote results/test_class_counts_last_cycle.csv")
    print(df.to_string(index=False))
    return df


def get_best_models():
    """Get deployed (CV-selected) model per subset from summary JSONs."""
    best = {}
    for fd in FDS:
        summ = json.load(open(RES / f"{fd}_summary.json"))
        best[fd] = summ["best_model"]
    return best


def compute_cis_for_model(fd: str, model_name: str, preds: dict):
    """Compute per-class precision/recall CIs for 3-class and binary."""
    rows = []
    units = preds["units"]

    # 3-class direct
    y3_true = preds["y3_true"]
    y3_pred = preds["y3_pred_direct"]
    for i, cname in enumerate(CLASS3_LABELS):
        # Precision
        mask_pred = (y3_pred == i)
        if mask_pred.sum() > 0:
            tp = ((y3_pred == i) & (y3_true == i)).sum()
            prec = tp / mask_pred.sum()
            lo, hi = wilson_ci(int(tp), int(mask_pred.sum()))
            rows.append({"fd": fd, "model": model_name, "approach": "direct",
                         "task": "3class", "class": cname, "metric": "precision",
                         "value": float(prec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_pred.sum())})
        # Recall
        mask_true = (y3_true == i)
        if mask_true.sum() > 0:
            tp = ((y3_pred == i) & (y3_true == i)).sum()
            rec = tp / mask_true.sum()
            lo, hi = wilson_ci(int(tp), int(mask_true.sum()))
            rows.append({"fd": fd, "model": model_name, "approach": "direct",
                         "task": "3class", "class": cname, "metric": "recall",
                         "value": float(rec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_true.sum())})

    # 3-class regress-then-threshold
    y3_pred_rtt = preds["y3_pred_rtt"]
    for i, cname in enumerate(CLASS3_LABELS):
        mask_pred = (y3_pred_rtt == i)
        if mask_pred.sum() > 0:
            tp = ((y3_pred_rtt == i) & (y3_true == i)).sum()
            prec = tp / mask_pred.sum()
            lo, hi = wilson_ci(int(tp), int(mask_pred.sum()))
            rows.append({"fd": fd, "model": f"{model_name}-via-RUL", "approach": "regress-threshold",
                         "task": "3class", "class": cname, "metric": "precision",
                         "value": float(prec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_pred.sum())})
        mask_true = (y3_true == i)
        if mask_true.sum() > 0:
            tp = ((y3_pred_rtt == i) & (y3_true == i)).sum()
            rec = tp / mask_true.sum()
            lo, hi = wilson_ci(int(tp), int(mask_true.sum()))
            rows.append({"fd": fd, "model": f"{model_name}-via-RUL", "approach": "regress-threshold",
                         "task": "3class", "class": cname, "metric": "recall",
                         "value": float(rec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_true.sum())})

    # Binary direct
    yb_true = preds["yb_true"]
    yb_pred = preds["yb_pred_direct"]
    # Negative class (RUL > 30)
    mask_pred = (yb_pred == 0)
    if mask_pred.sum() > 0:
        tp = ((yb_pred == 0) & (yb_true == 0)).sum()
        prec = tp / mask_pred.sum()
        lo, hi = wilson_ci(int(tp), int(mask_pred.sum()))
        rows.append({"fd": fd, "model": model_name, "approach": "direct",
                     "task": "binary30", "class": "negative", "metric": "precision",
                     "value": float(prec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_pred.sum())})
    mask_true = (yb_true == 0)
    if mask_true.sum() > 0:
        tp = ((yb_pred == 0) & (yb_true == 0)).sum()
        rec = tp / mask_true.sum()
        lo, hi = wilson_ci(int(tp), int(mask_true.sum()))
        rows.append({"fd": fd, "model": model_name, "approach": "direct",
                     "task": "binary30", "class": "negative", "metric": "recall",
                     "value": float(rec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_true.sum())})
    # Positive class (RUL <= 30)
    mask_pred = (yb_pred == 1)
    if mask_pred.sum() > 0:
        tp = ((yb_pred == 1) & (yb_true == 1)).sum()
        prec = tp / mask_pred.sum()
        lo, hi = wilson_ci(int(tp), int(mask_pred.sum()))
        rows.append({"fd": fd, "model": model_name, "approach": "direct",
                     "task": "binary30", "class": "positive", "metric": "precision",
                     "value": float(prec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_pred.sum())})
    mask_true = (yb_true == 1)
    if mask_true.sum() > 0:
        tp = ((yb_pred == 1) & (yb_true == 1)).sum()
        rec = tp / mask_true.sum()
        lo, hi = wilson_ci(int(tp), int(mask_true.sum()))
        rows.append({"fd": fd, "model": model_name, "approach": "direct",
                     "task": "binary30", "class": "positive", "metric": "recall",
                     "value": float(rec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_true.sum())})

    # Binary regress-then-threshold
    yb_pred_rtt = preds["yb_pred_rtt"]
    mask_pred = (yb_pred_rtt == 1)
    if mask_pred.sum() > 0:
        tp = ((yb_pred_rtt == 1) & (yb_true == 1)).sum()
        prec = tp / mask_pred.sum()
        lo, hi = wilson_ci(int(tp), int(mask_pred.sum()))
        rows.append({"fd": fd, "model": f"{model_name}-via-RUL", "approach": "regress-threshold",
                     "task": "binary30", "class": "positive", "metric": "precision",
                     "value": float(prec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_pred.sum())})
    mask_true = (yb_true == 1)
    if mask_true.sum() > 0:
        tp = ((yb_pred_rtt == 1) & (yb_true == 1)).sum()
        rec = tp / mask_true.sum()
        lo, hi = wilson_ci(int(tp), int(mask_true.sum()))
        rows.append({"fd": fd, "model": f"{model_name}-via-RUL", "approach": "regress-threshold",
                     "task": "binary30", "class": "positive", "metric": "recall",
                     "value": float(rec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_true.sum())})
    mask_pred = (yb_pred_rtt == 0)
    if mask_pred.sum() > 0:
        tp = ((yb_pred_rtt == 0) & (yb_true == 0)).sum()
        prec = tp / mask_pred.sum()
        lo, hi = wilson_ci(int(tp), int(mask_pred.sum()))
        rows.append({"fd": fd, "model": f"{model_name}-via-RUL", "approach": "regress-threshold",
                     "task": "binary30", "class": "negative", "metric": "precision",
                     "value": float(prec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_pred.sum())})
    mask_true = (yb_true == 0)
    if mask_true.sum() > 0:
        tp = ((yb_pred_rtt == 0) & (yb_true == 0)).sum()
        rec = tp / mask_true.sum()
        lo, hi = wilson_ci(int(tp), int(mask_true.sum()))
        rows.append({"fd": fd, "model": f"{model_name}-via-RUL", "approach": "regress-threshold",
                     "task": "binary30", "class": "negative", "metric": "recall",
                     "value": float(rec), "ci_lo": lo, "ci_hi": hi, "support": int(mask_true.sum())})

    return rows


def main():
    # 1. Last-cycle class counts
    compute_last_cycle_counts()

    # 2. Per-class CIs for best model + main baselines
    best_models = get_best_models()
    baselines = ["logreg", "rf", "hgb"]
    all_rows = []

    for fd in FDS:
        models_to_eval = [best_models[fd]] + [m for m in baselines if m != best_models[fd]]
        for model_name in models_to_eval:
            print(f"Computing CIs for {fd} {model_name}...")
            preds = get_predictions_for_model(fd, model_name)
            rows = compute_cis_for_model(fd, model_name, preds)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(RES / "test_classification_cis_last_cycle.csv", index=False)
    print(f"\nWrote results/test_classification_cis_last_cycle.csv ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()