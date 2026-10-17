"""Training: grouped CV, tuning, test evaluation, ablations."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from .data import (SENSORS, add_class_labels, add_piecewise_rul, add_rul_test_all_cycles,
                   add_rul_train, dataset_summary, load_split)
from .evaluate import (binary_metrics, class_metrics_3, lead_time_analysis,
                       false_alarm_rate, regression_metrics)
from .models import TORCH_OK, get_classifiers, get_regressors
from .preprocess import (ConditionNormalizer, feature_matrix, make_sequences,
                         select_sensors_by_variance, smooth_per_unit)

log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_subset(fd: str, raw_dir: str, cfg: dict, rul_cap: float | None = None):
    cap = rul_cap or cfg["rul_cap"]
    train, test, rul_last = load_split(raw_dir, fd)
    train = add_rul_train(train)
    test_all = add_rul_test_all_cycles(test, rul_last)
    train = add_piecewise_rul(train, cap)
    test_all = add_piecewise_rul(test_all, cap)
    train = add_class_labels(train)
    test_all = add_class_labels(test_all)
    return train, test_all, rul_last


def build_feature_store(train, test_all, cfg, fd, mode_override=None, windows_override=None,
                        smooth=True, norm_mode_override=None):
    """Fixed order (fits on TRAIN only): condition-normalise ALL sensors first,
    then variance-drop on the normalised train frame, then causal smoothing per
    unit on normalised sensors, then causal rolling features + train-fit scaler.

    Rationale: on FD002/FD004 consecutive cycles can sit in different operating
    conditions, so smoothing raw sensors blends conditions and a later z-score
    cannot undo it. Normalising first keeps within-condition dynamics only.
    """
    from .data import SENSORS
    windows = windows_override or cfg["rolling_windows"]
    mode = norm_mode_override or mode_override or ("cluster" if fd in ("FD002", "FD004") else "global")
    # 1. condition normalisation on ALL sensors (fit: train only)
    norm = ConditionNormalizer(mode=mode, k=cfg.get("kmeans_k", 6),
                               random_state=cfg.get("seed", 42)).fit(train, list(SENSORS))
    train_n = norm.transform(train)
    test_n = norm.transform(test_all)
    # 2. variance / near-constant drop on the NORMALISED train frame
    keep, drop = select_sensors_by_variance(train_n, list(SENSORS), cfg.get("variance_threshold", 1e-6))
    log.info("%s keep=%d drop=%s", fd, len(keep), drop)
    use = [c for c in keep]
    # 3. causal exponential smoothing per unit on normalised sensors
    if smooth:
        alpha = cfg.get("smooth_alpha", 0.3)
        train_s = smooth_per_unit(train_n, use, alpha)
        test_s = smooth_per_unit(test_n, use, alpha)
        for c in use:  # index-aligned assignment (frames were row-sorted by the smoother)
            train_n[c] = train_s[c + "_sm"]
            test_n[c] = test_s[c + "_sm"]
    # re-add labels lost? transform preserves all columns
    for col in ["RUL", "RUL_cap", "RUL_last", "cls3", "cls3_id", "bin30", "unit", "cycle"]:
        if col in train.columns and col not in train_n.columns:
            train_n[col] = train[col].values
        if col in test_all.columns and col not in test_n.columns:
            test_n[col] = test_all[col].values
    Xtr, feat_cols = feature_matrix(train_n, use, windows)
    Xte, _ = feature_matrix(test_n, use, windows)
    scaler = StandardScaler().fit(Xtr.values)
    Xtr_s = scaler.transform(Xtr.values)
    Xte_s = scaler.transform(Xte.values)
    return {"Xtr": Xtr_s, "Xte": Xte_s, "feat_cols": feat_cols, "sensors": use,
            "dropped": drop, "train": train_n, "test": test_n, "scaler": scaler, "norm": norm}


def grouped_cv_regression(X, y, groups, models, n_splits=5, seed=42):
    """Grouped K-fold CV (by engine) for regressors on capped RUL.

    Returns per-model mean±std of RMSE, MAE and the PHM08 score.
    Model selection must use these scores, never the test set.
    """
    from .evaluate import phm_score, rmse
    from sklearn.metrics import mean_absolute_error
    gkf = GroupKFold(n_splits=n_splits)
    out = {}
    for name, mdl in models.items():
        rms, maes, phms = [], [], []
        for tr, va in gkf.split(X, y, groups):
            mdl.fit(X[tr], y[tr])
            p = mdl.predict(X[va])
            rms.append(rmse(y[va], p))
            maes.append(mean_absolute_error(y[va], p))
            phms.append(phm_score(y[va], p))
        out[name] = {"cv_rmse_mean": float(np.mean(rms)), "cv_rmse_std": float(np.std(rms)),
                     "cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
                     "cv_phm_mean": float(np.mean(phms)), "cv_phm_std": float(np.std(phms))}
    return out


def grouped_cv_classification(X, y, groups, models, n_splits=5, seed=42):
    """Grouped K-fold CV for classifiers. Score: macro-F1 + balanced accuracy."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    gkf = GroupKFold(n_splits=n_splits)
    out = {}
    for name, mdl in models.items():
        f1s, bas = [], []
        for tr, va in gkf.split(X, y, groups):
            mdl.fit(X[tr], y[tr])
            p = mdl.predict(X[va])
            f1s.append(f1_score(y[va], p, average="macro", zero_division=0))
            bas.append(balanced_accuracy_score(y[va], p))
        out[name] = {"cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s)),
                     "cv_bal_acc_mean": float(np.mean(bas)), "cv_bal_acc_std": float(np.std(bas))}
    return out


def oof_predict(make_model, X, y, groups, n_splits=5, seed=42):
    """Out-of-fold predictions on the training set (grouped by engine).

    `make_model` is a zero-arg factory returning a fresh estimator.
    Returns array aligned with X. Used for threshold tuning and conformal
    calibration so the test set is never touched.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(X))
    for tr, va in gkf.split(X, y, groups):
        mdl = make_model()
        mdl.fit(X[tr], y[tr])
        oof[va] = mdl.predict(X[va])
    return oof


def tune_hgb_regressor(X, y, groups, seed=42, n_iter=12, n_splits=5):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import loguniform, randint
    param = {"max_iter": randint(100, 400), "max_depth": [None, 5, 10, 15],
             "learning_rate": loguniform(0.02, 0.3), "l2_regularization": loguniform(1e-3, 10.0)}
    gkf = GroupKFold(n_splits=n_splits)
    rs = RandomizedSearchCV(HistGradientBoostingRegressor(random_state=seed), param,
                            n_iter=n_iter, scoring="neg_root_mean_squared_error",
                            cv=list(gkf.split(X, y, groups)), random_state=seed, n_jobs=1)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_, float(-rs.best_score_)


def rul_to_classes(rul: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(rul, dtype=float)
    c3 = np.where(r > 50, 0, np.where(r >= 20, 1, 2))
    b = (r <= 30).astype(int)
    return c3, b
