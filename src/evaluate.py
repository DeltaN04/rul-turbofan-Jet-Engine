"""Metrics: RMSE/MAE/R2, PHM08 score, classification, lead-time, cost."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, r2_score,
                             roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score)


def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def phm_score(y_true, y_pred) -> float:
    """NASA PHM08 asymmetric score. d = pred - true."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {"rmse": rmse(y_true, y_pred), "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)), "phm": float(phm_score(y_true, y_pred)),
            "n": int(len(y_true))}


def class_metrics_3(y_true_id: np.ndarray, y_pred_id: np.ndarray) -> dict:
    labels = [0, 1, 2]
    names = ["safe", "warning", "critical"]
    out: dict = {}
    out["accuracy"] = float(accuracy_score(y_true_id, y_pred_id))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true_id, y_pred_id))
    out["mcc"] = float(matthews_corrcoef(y_true_id, y_pred_id)) if len(np.unique(y_true_id)) > 1 else 0.0
    out["kappa"] = float(cohen_kappa_score(y_true_id, y_pred_id))
    cm = confusion_matrix(y_true_id, y_pred_id, labels=labels)
    out["confusion_matrix"] = cm
    for i, nm in enumerate(names):
        tp = cm[i, i]
        out[f"precision_{nm}"] = float(tp / cm[:, i].sum()) if cm[:, i].sum() else 0.0
        out[f"recall_{nm}"] = float(tp / cm[i, :].sum()) if cm[i, :].sum() else 0.0
        p, r = out[f"precision_{nm}"], out[f"recall_{nm}"]
        out[f"f1_{nm}"] = float(2 * p * r / (p + r)) if (p + r) else 0.0
        out[f"support_{nm}"] = int(cm[i, :].sum())
    prec = [out[f"precision_{n}"] for n in names]
    rec = [out[f"recall_{n}"] for n in names]
    f1 = [out[f"f1_{n}"] for n in names]
    out["macro_precision"] = float(np.mean(prec))
    out["macro_recall"] = float(np.mean(rec))
    out["macro_f1"] = float(np.mean(f1))
    sup = np.array([out[f"support_{n}"] for n in names], dtype=float)
    w = sup / sup.sum() if sup.sum() else sup
    out["weighted_precision"] = float(np.dot(prec, w))
    out["weighted_recall"] = float(np.dot(rec, w))
    out["weighted_f1"] = float(np.dot(f1, w))
    return out


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    out: dict = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["mcc"] = 0.0
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def expected_cost(cm2: np.ndarray, missed: float = 10.0, false_alarm: float = 1.0) -> float:
    """cm rows=true[neg,pos], cols=pred[neg,pos]. cost = FN*missed + FP*false_alarm."""
    tn, fp, fn, tp = cm2[0, 0], cm2[0, 1], cm2[1, 0], cm2[1, 1]
    return float(fn * missed + fp * false_alarm)


def lead_time_analysis(test_df: pd.DataFrame, pred_rul_per_row: np.ndarray,
                       warn_at: float = 50.0, crit_at: float = 20.0) -> pd.DataFrame:
    """First cycle each test engine crosses warning/critical on PREDICTED RUL,
    and the resulting lead time (true cycles remaining at that point)."""
    df = test_df[["unit", "cycle", "RUL"]].copy()
    df["pred"] = np.asarray(pred_rul_per_row, dtype=float)
    rows = []
    for u, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        g = g.reset_index(drop=True)
        first_w = g.index[g["pred"] <= warn_at]
        first_c = g.index[g["pred"] <= crit_at]
        def lead(idx):
            if len(idx) == 0:
                return np.nan, np.nan
            k = int(idx[0])
            # cycles before true failure when alert raised = true RUL at that row
            return int(g.loc[k, "cycle"]), float(g.loc[k, "RUL"])
        wc, wl = lead(first_w)
        cc, cl = lead(first_c)
        rows.append({"unit": int(u), "warn_cycle": wc, "warn_lead": wl,
                     "crit_cycle": cc, "crit_lead": cl,
                     "n_cycles": int(len(g)), "true_last_rul": float(g["RUL"].iloc[-1])})
    return pd.DataFrame(rows)


def false_alarm_rate(test_df: pd.DataFrame, pred_rul_per_row: np.ndarray,
                     safe_rul: float = 50.0) -> float:
    """False alarms per 100 cycles: predicted warning/critical while true RUL > 50."""
    df = test_df[["RUL"]].copy()
    df["pred"] = np.asarray(pred_rul_per_row, dtype=float)
    false = ((df["pred"] <= 50) & (df["RUL"] > safe_rul)).sum()
    return float(false / len(df) * 100.0)


def bootstrap_ci_by_engine(units, y_true, y_pred, metric_fn, n_boot: int = 1000,
                           seed: int = 42, ci: float = 95.0) -> tuple[float, float, float]:
    """Bootstrap 95% CI resampling ENGINES (not rows) with replacement."""
    rng = np.random.RandomState(seed)
    units = np.asarray(units)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    uniq = np.unique(units)
    idx_by_unit = {u: np.where(units == u)[0] for u in uniq}
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_unit[u] for u in samp])
        vals.append(float(metric_fn(y_true[idx], y_pred[idx])))
    lo, hi = np.percentile(vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(np.mean(vals)), float(lo), float(hi)


def binned_error_table(y_true, y_pred, bins=(0, 20, 50, 125)) -> list[dict]:
    """RMSE/MAE per true-RUL bin (last bin inclusive of the cap)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (y_true >= lo) & (y_true <= hi if i == len(bins) - 2 else y_true < hi)
        if m.sum() == 0:
            rows.append({"bin": f"{lo}-{hi}", "rmse": float("nan"), "mae": float("nan"), "n": 0})
        else:
            rows.append({"bin": f"{lo}-{hi}", "rmse": rmse(y_true[m], y_pred[m]),
                         "mae": float(mean_absolute_error(y_true[m], y_pred[m])), "n": int(m.sum())})
    return rows
