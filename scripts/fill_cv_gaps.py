"""Backfill CV scores the main run didn't store (deterministic, same seeds/folds).

- hgb_tuned regressor: grouped 5-fold CV with the tuned best_params from summary.
- mean (DummyRegressor): grouped 5-fold CV.
- direct classifiers: grouped 5-fold CV macro-F1 on 3-class + binary tasks.
Patches results/{fd}_regression.csv + master_regression.csv and writes
results/{fd}_clf_cv.json; merges classifier-CV headlines into summary JSONs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import (build_feature_store, grouped_cv_classification,
                       grouped_cv_regression, prepare_subset)

SEED = 42


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    n_splits = int(cfg.get("n_splits_cv", 5))
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"== {fd} ==", flush=True)
        summ = json.load(open(f"results/{fd}_summary.json"))
        train, test_all, _ = prepare_subset(fd, "data/raw", cfg)
        store = build_feature_store(train.copy(), test_all.copy(), cfg, fd)
        Xtr = store["Xtr"]
        groups = train["unit"].values
        ycap = train["RUL_cap"].values
        # 1. hgb_tuned + mean CV
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        extra = {"hgb_tuned": HistGradientBoostingRegressor(
            **summ["best_hgb_params"], random_state=SEED),
            "mean": DummyRegressor(strategy="mean")}
        cvr = grouped_cv_regression(Xtr, ycap, groups, extra, n_splits, SEED)
        reg = pd.read_csv(f"results/{fd}_regression.csv")
        for name, v in cvr.items():
            m = reg.model == name
            reg.loc[m, "cv_rmse"] = v["cv_rmse_mean"]
            reg.loc[m, "cv_rmse_std"] = v["cv_rmse_std"]
            reg.loc[m, "cv_phm"] = v["cv_phm_mean"]
            reg.loc[m, "cv_phm_std"] = v["cv_phm_std"]
            reg.loc[m, "cv_note"] = f"grouped {n_splits}-fold CV"
        reg.to_csv(f"results/{fd}_regression.csv", index=False)
        # 2. classifier CV on both tasks
        from src.models import get_classifiers
        clfs = get_classifiers(seed=SEED, fast=False)
        cv3 = grouped_cv_classification(Xtr, train["cls3_id"].values, groups, clfs,
                                        n_splits, SEED)
        cvb = grouped_cv_classification(Xtr, train["bin30"].values, groups, clfs,
                                        n_splits, SEED)
        head3 = max(cv3, key=lambda k: cv3[k]["cv_macro_f1_mean"])
        headb = max(cvb, key=lambda k: cvb[k]["cv_macro_f1_mean"])
        json.dump({"cv3": cv3, "cvb": cvb, "head3": head3, "headb": headb},
                  open(f"results/{fd}_clf_cv.json", "w"), indent=2)
        summ["clf_cv"] = {"cv3": cv3, "cvb": cvb, "head3": head3, "headb": headb}
        json.dump(summ, open(f"results/{fd}_summary.json", "w"), indent=2, default=str)
        print(f"{fd}: head3={head3} headb={headb}", flush=True)
    m = pd.concat([pd.read_csv(f"results/{fd}_regression.csv")
                   for fd in ["FD001", "FD002", "FD003", "FD004"]])
    m.to_csv("results/master_regression.csv", index=False)
    print("patched master_regression.csv")


if __name__ == "__main__":
    main()
