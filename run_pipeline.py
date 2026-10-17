"""End-to-end pipeline: EDA + features + regression + classification + ablations.

Usage: python run_pipeline.py  (from aucml_rul_project/)
Saves figures/ results/ models/. All seeded.
"""
from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")

from src.data import SENSORS, OPS, add_class_labels, dataset_summary  # noqa: E402
from src.train import (build_feature_store, grouped_cv_classification, grouped_cv_regression,
                         oof_predict, prepare_subset, rul_to_classes)  # noqa: E402
from src.evaluate import (binary_metrics, class_metrics_3, expected_cost,  # noqa: E402
                          false_alarm_rate, lead_time_analysis, regression_metrics)
from src.models import TORCH_OK, get_classifiers, get_regressors  # noqa: E402
from src.plots import save, setup  # noqa: E402
from src.explain import permutation_top  # noqa: E402

setup()
SEED = 42
np.random.seed(SEED)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def ensure_dirs():
    for d in ["figures", "results", "models", "data/processed"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------- EDA ----------------
def eda_subset(fd, train, test_all, rul_last):
    s = dataset_summary(train, test_all, rul_last)
    # 1. lifetime distribution
    life = train.groupby("unit")["cycle"].max()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(life.values, bins=25, color="#0173B2", edgecolor="white")
    ax.axvline(life.mean(), color="#DE8F05", linestyle="--", label=f"mean={life.mean():.0f}")
    ax.set_title(f"{fd}: engine lifetime distribution (train, run-to-failure)")
    ax.set_xlabel("cycles to failure"); ax.set_ylabel("engines")
    ax.legend()
    save(fig, f"figures/{fd}_lifetime_hist.png")
    # 2. sensor trajectories aligned by RUL for 3 engines
    units = sorted(train["unit"].unique())[:3]
    sel = ["s2", "s3", "s4", "s7", "s11", "s12"]
    sel = [c for c in sel if c in train.columns]
    fig, axes = plt.subplots(len(sel), 1, figsize=(8, 2 * len(sel)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), sel):
        for u in units:
            g = train[train.unit == u].sort_values("cycle")
            ax.plot(g["RUL"].values, g[c].values, label=f"unit {u}", linewidth=1.2)
        ax.set_ylabel(c)
    axes.flat[-1].set_xlabel("true RUL (cycles to failure)")
    axes.flat[0].set_title(f"{fd}: sensor trajectories vs RUL (3 engines)")
    axes.flat[0].legend(fontsize=8)
    save(fig, f"figures/{fd}_trajectories.png")
    # 3. correlation heatmap sensor vs RUL
    corr = train[SENSORS + ["RUL"]].corr(numeric_only=True)["RUL"].drop("RUL").sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    corr.plot.barh(ax=ax, color="#0173B2")
    ax.set_title(f"{fd}: sensor correlation with RUL")
    ax.set_xlabel("Pearson r")
    save(fig, f"figures/{fd}_corr_rul.png")
    # 4. monotonicity bar
    mono = {}
    for c in SENSORS:
        m = 0.0
        for _, g in train.groupby("unit"):
            gg = g.sort_values("cycle")
            d = np.diff(gg[c].values)
            if len(d):
                mono[c] = mono.get(c, 0.0) + abs(np.mean(np.sign(d)))
        mono[c] = mono.get(c, 0.0) / train["unit"].nunique()
    ms = pd.Series(mono).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ms.plot.barh(ax=ax, color="#029E73")
    ax.set_title(f"{fd}: sensor monotonicity (mean |sign of step| per engine)")
    ax.set_xlabel("monotonicity")
    save(fig, f"figures/{fd}_monotonicity.png")
    # 5. class distribution
    cls = train["cls3"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    cls.reindex(["safe", "warning", "critical"]).plot.bar(ax=ax, color=["#029E73", "#DE8F05", "#CC78BC"])
    ax.set_title(f"{fd}: class distribution (train rows)")
    ax.set_ylabel("rows")
    for i, v in enumerate(cls.reindex(["safe", "warning", "critical"]).values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom")
    save(fig, f"figures/{fd}_classdist.png")
    # 6. raw vs smoothed
    u0 = sorted(train["unit"].unique())[0]
    g = train[train.unit == u0].sort_values("cycle")
    c0 = "s11" if "s11" in train.columns else SENSORS[0]
    sm = g[c0].ewm(alpha=0.3, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(g["cycle"], g[c0], label="raw", alpha=0.6, linewidth=1)
    ax.plot(g["cycle"], sm, label="exp-smoothed (α=0.3)", linewidth=1.5)
    ax.set_title(f"{fd}: noise illustration — {c0} for unit {u0}")
    ax.set_xlabel("cycle"); ax.set_ylabel(c0); ax.legend()
    save(fig, f"figures/{fd}_smoothing.png")
    return s


def condition_cluster_plots(fd, train, test_all, cfg):
    """KMeans on standardised op settings; before/after normalisation (FD002/FD004).

    Before/after panels use a degradation-informative sensor (s11), not the
    condition proxy s1; a second panel shows within-condition scatter.
    EDA trajectories are also drawn on normalised sensors for these subsets.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    if fd not in ("FD002", "FD004"):
        return {"k": 1, "silhouette": float("nan")}
    Xs = StandardScaler().fit_transform(train[OPS].values)
    inertias, sils = [], []
    for k in [2, 3, 4, 5, 6, 8]:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, km.labels_) if k > 1 else float("nan"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([2, 3, 4, 5, 6, 8], inertias, marker="o")
    axes[0].set_title(f"{fd}: KMeans elbow on standardised operating settings")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia")
    axes[1].plot([2, 3, 4, 5, 6, 8], sils, marker="o", color="#DE8F05")
    axes[1].set_title(f"{fd}: silhouette vs k (k=6 justified)")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette")
    save(fig, f"figures/{fd}_kmeans_select.png")
    # before/after on an informative sensor (s11), plus within-condition scatter
    from src.preprocess import ConditionNormalizer
    from src.data import SENSORS as _S
    norm = ConditionNormalizer(mode="cluster", k=6, random_state=SEED).fit(train, list(_S))
    trn = norm.transform(train)
    lab = norm.kmeans.labels_
    c = "s11" if "s11" in trn.columns else store_sensors_fallback(train)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(train[c].values, bins=40, color="#0173B2")
    axes[0].set_title(f"{fd}: {c} before normalisation (6 modes mixed)")
    axes[0].set_xlabel(c); axes[0].set_ylabel("count")
    axes[1].hist(trn[c].values, bins=40, color="#029E73")
    axes[1].set_title(f"{fd}: {c} after within-cluster z-score")
    axes[1].set_xlabel(c); axes[1].set_ylabel("count")
    save(fig, f"figures/{fd}_condnorm_effect.png")
    # within-condition scatter: s11 vs RUL coloured by cluster (normalised)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sc = axes[0].scatter(train["RUL"].values[::10], train[c].values[::10], c=lab[::10],
                         cmap="tab10", s=6, alpha=0.7)
    axes[0].set_title(f"{fd}: raw {c} vs RUL by condition (every 10th row)")
    axes[0].set_xlabel("true RUL (cycles)"); axes[0].set_ylabel(f"raw {c}")
    fig.colorbar(sc, ax=axes[0], label="cluster")
    sc2 = axes[1].scatter(trn["RUL"].values[::10], trn[c].values[::10], c=lab[::10],
                          cmap="tab10", s=6, alpha=0.7)
    axes[1].set_title(f"{fd}: normalised {c} vs RUL (condition effect removed)")
    axes[1].set_xlabel("true RUL (cycles)"); axes[1].set_ylabel(f"normalised {c}")
    fig.colorbar(sc2, ax=axes[1], label="cluster")
    save(fig, f"figures/{fd}_condnorm_scatter.png")
    # EDA trajectories on NORMALISED sensors for multi-condition subsets
    units = sorted(trn["unit"].unique())[:3]
    sel = ["s4", "s7", "s11", "s12"]
    sel = [x for x in sel if x in trn.columns]
    fig, axes = plt.subplots(len(sel), 1, figsize=(8, 2 * len(sel)), sharex=True)
    for ax, cc in zip(np.atleast_1d(axes), sel):
        for u in units:
            g = trn[trn.unit == u].sort_values("cycle")
            ax.plot(g["RUL"].values, g[cc].values, label=f"unit {u}", linewidth=1.2)
        ax.set_ylabel(f"{cc} (norm)")
    axes.flat[-1].set_xlabel("true RUL (cycles to failure)")
    axes.flat[0].set_title(f"{fd}: normalised sensor trajectories vs RUL (3 engines)")
    axes.flat[0].legend(fontsize=8)
    save(fig, f"figures/{fd}_trajectories_norm.png")
    km = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(Xs)
    return {"k": 6, "silhouette": float(silhouette_score(Xs, km.labels_))}


def store_sensors_fallback(train):
    from src.data import SENSORS as _S
    keep = [c for c in _S if float(train[c].var()) > 1e-6]
    return keep[0] if keep else _S[0]


# ---------------- deep learning ----------------
def train_torch_model(X_seq, y_seq, unit_ids, n_feat, kind="lstm", epochs=60, patience=8,
                      seed=42, hidden=48, batch=512):
    """Train one sequence model with a BY-ENGINE validation split.

    Returns (model, mu, sd, val_rmse) where val_rmse is the holdout RMSE on
    capped RUL (used as this model's CV-like score for model selection).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import GroupShuffleSplit
    from src.models import CNN1D, LSTMReg
    torch.manual_seed(seed)
    np.random.seed(seed)
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
    tri, vai = next(gss.split(X_seq, y_seq, unit_ids))
    mu = X_seq[tri].reshape(-1, n_feat).mean(0)
    sd = X_seq[tri].reshape(-1, n_feat).std(0) + 1e-8
    def _std(a):
        return ((a.reshape(-1, n_feat) - mu) / sd).reshape(a.shape).astype(np.float32)
    tl = DataLoader(TensorDataset(torch.from_numpy(_std(X_seq[tri])),
                                  torch.from_numpy(y_seq[tri].astype(np.float32))),
                    batch_size=batch, shuffle=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(_std(X_seq[vai])),
                                  torch.from_numpy(y_seq[vai].astype(np.float32))),
                    batch_size=1024)
    if kind == "lstm":
        model = LSTMReg(n_feat, hidden=hidden, layers=1)
    elif kind == "cnn":
        model = CNN1D(n_feat)
    else:
        raise ValueError(kind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = torch.nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([loss_fn(model(xb), yb).item() for xb, yb in vl]))
        sched.step(v)
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: vv.cpu() for k, vv in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, mu.astype(np.float32), sd.astype(np.float32), float(np.sqrt(best))


CLASS3_LABELS = ["safe", "warning", "critical"]
BIN_LABELS = ["neg(≥31)", "pos(≤30)"]


def tidy_rows_3class(fd, approach, model, y_true, y_pred):
    """One row per class + macro + weighted (A5 tidy format)."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                                 confusion_matrix, matthews_corrcoef,
                                 precision_recall_fscore_support)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    kap = float(cohen_kappa_score(y_true, y_pred))
    tot = float(sup.sum()) if sup.sum() else 1.0
    agg = {"macro": (float(prec.mean()), float(rec.mean()), float(f1.mean())),
           "weighted": (float(np.dot(prec, sup) / tot), float(np.dot(rec, sup) / tot),
                        float(np.dot(f1, sup) / tot))}
    rows = []
    for i, cname in enumerate(CLASS3_LABELS):
        rows.append({"fd": fd, "task": "3class", "approach": approach, "model": model,
                     "class": cname, "precision": float(prec[i]), "recall": float(rec[i]),
                     "f1": float(f1[i]), "support": int(sup[i]),
                     "accuracy": acc, "balanced_acc": bal, "mcc": mcc, "kappa": kap,
                     "roc_auc": float("nan"), "pr_auc": float("nan")})
    for aname, (p, r, f) in agg.items():
        rows.append({"fd": fd, "task": "3class", "approach": approach, "model": model,
                     "class": aname, "precision": p, "recall": r, "f1": f,
                     "support": int(len(y_true)),
                     "accuracy": acc, "balanced_acc": bal, "mcc": mcc, "kappa": kap,
                     "roc_auc": float("nan"), "pr_auc": float("nan")})
    return rows, confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def tidy_rows_binary(fd, approach, model, y_true, y_pred, y_score):
    """Rows: negative / positive / macro. roc/pr filled from scores."""
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix, matthews_corrcoef,
                                 precision_recall_fscore_support, roc_auc_score)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    kap = float(cohen_kappa_score(y_true, y_pred))
    if y_score is not None and len(np.unique(y_true)) > 1:
        roc = float(roc_auc_score(y_true, y_score))
        pra = float(average_precision_score(y_true, y_score))
    else:
        roc, pra = float("nan"), float("nan")
    rows = []
    for i, cname in enumerate(["negative", "positive"]):
        rows.append({"fd": fd, "task": "binary30", "approach": approach, "model": model,
                     "class": cname, "precision": float(prec[i]), "recall": float(rec[i]),
                     "f1": float(f1[i]), "support": int(sup[i]),
                     "accuracy": acc, "balanced_acc": bal, "mcc": mcc, "kappa": kap,
                     "roc_auc": roc, "pr_auc": pra})
    rows.append({"fd": fd, "task": "binary30", "approach": approach, "model": model,
                 "class": "macro", "precision": float(prec.mean()), "recall": float(rec.mean()),
                 "f1": float(f1.mean()), "support": int(len(y_true)),
                 "accuracy": acc, "balanced_acc": bal, "mcc": mcc, "kappa": kap,
                 "roc_auc": roc, "pr_auc": pra})
    return rows, confusion_matrix(y_true, y_pred, labels=[0, 1])


def plot_labeled_cm(cm, display_labels, title, path, normalize=False):
    """Confusion matrix with class-name axes, annotated counts and colorbar."""
    fig, ax = plt.subplots(figsize=(5, 4))
    a = np.asarray(cm, dtype=float)
    if normalize:
        a = a / a.sum(axis=1, keepdims=True).clip(min=1e-9)
    im = ax.imshow(a, cmap="Blues", vmin=0, vmax=(1.0 if normalize else None))
    ax.set_xticks(range(len(display_labels))); ax.set_xticklabels(display_labels)
    ax.set_yticks(range(len(display_labels))); ax.set_yticklabels(display_labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fmt = ".2f" if normalize else ".0f"
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a[i, j]
            thr = 0.5 if normalize else (a.max() / 2 if a.max() else 1)
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                    color="white" if v > thr else "black")
    fig.colorbar(im, ax=ax)
    save(fig, path)


# ---------------- per-subset run ----------------
def run_subset(fd, cfg, raw_dir="data/raw", do_deep=True, n_iter=8, smoke=False):
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import GroupKFold
    from src.preprocess import make_sequences
    from src.train import tune_hgb_regressor
    log.info("=== %s ===", fd)
    train, test_all, rul_last = prepare_subset(fd, raw_dir, cfg)
    summ = eda_subset(fd, train, test_all, rul_last)
    summ.update(condition_cluster_plots(fd, train, test_all, cfg))
    summ["fd"] = fd
    store = build_feature_store(train.copy(), test_all.copy(), cfg, fd)
    Xtr, Xte = store["Xtr"], store["Xte"]
    feat_cols = store["feat_cols"]
    ytr_cap = train["RUL_cap"].values
    ytr_true = train["RUL"].values
    yte_true_all = test_all["RUL"].values
    yte_cap_all = test_all["RUL_cap"].values
    groups = train["unit"].values
    # last-cycle masks
    last_idx = test_all.groupby("unit")["cycle"].idxmax().sort_values()
    # map idxmax labels -> positional indices
    pos = {ix: i for i, ix in enumerate(test_all.index)}
    last_pos = np.array([pos[i] for i in last_idx.values])
    yte_true_last = yte_true_all[last_pos]
    yte_cap_last = yte_cap_all[last_pos]
    Xte_last = Xte[last_pos]

    # ---- regression: grouped 5-fold CV for EVERY sklearn model, full configs (A2/A3) ----
    n_splits = int(cfg.get("n_splits_cv", 5))
    seeds = list(cfg.get("seeds", [42, 43, 44])) if not smoke else [SEED]
    regs = get_regressors(seed=SEED, fast=smoke)
    cvres = grouped_cv_regression(Xtr, ytr_cap, groups, regs, n_splits=n_splits, seed=SEED)
    # tune HGB (grouped 5-fold)
    try:
        best_hgb, best_params, best_cv = tune_hgb_regressor(Xtr, ytr_cap, groups, SEED, n_iter, n_splits)
    except Exception as e:
        log.warning("tuning failed %s: %s", fd, e)
        from sklearn.ensemble import HistGradientBoostingRegressor
        best_hgb, best_params, best_cv = HistGradientBoostingRegressor(random_state=SEED), {}, float("nan")
    # final fits on full train: single seed, except rf/mlp (3 seeds, A10)
    fitted, seed_test_preds = {}, {}
    multi_seed = {"rf", "mlp"}
    for name in regs:
        if name in multi_seed and len(seeds) > 1:
            arrs = []
            for s in seeds:
                mdl = get_regressors(seed=s, fast=smoke)[name]
                mdl.fit(Xtr, ytr_cap)
                arrs.append(mdl.predict(Xte_last))
            seed_test_preds[name] = arrs
            fitted[name] = get_regressors(seed=SEED, fast=smoke)[name].fit(Xtr, ytr_cap)
        else:
            mdl = regs[name].fit(Xtr, ytr_cap)
            fitted[name] = mdl
            seed_test_preds[name] = [mdl.predict(Xte_last)]
    fitted["hgb_tuned"] = best_hgb
    best_hgb.fit(Xtr, ytr_cap)
    seed_test_preds["hgb_tuned"] = [best_hgb.predict(Xte_last)]
    # dummy baseline
    dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr_cap)
    fitted["mean"] = dummy
    seed_test_preds["mean"] = [dummy.predict(Xte_last)]

    # deep models on sequences from the FIXED pipeline (A1), on ALL subsets (A4)
    deep_info, deep_holdout = {}, {}
    deep_epochs = 3 if smoke else int(cfg.get("deep_epochs", 60))
    deep_patience = int(cfg.get("deep_patience", 8))
    deep_seeds = [SEED] if smoke else seeds
    torch_allrow = {}  # kind -> all-window test preds (seed42) for trajectories
    if do_deep and TORCH_OK:
        import torch
        from src.preprocess import make_sequences as _ms
        seq_cols = store["sensors"]
        trn, ten = store["train"], store["test"]
        Xs_tr, ys_tr, us_tr = _ms(trn, seq_cols, window=cfg["window_length"])
        Xs_te, _, _ = _ms(ten, seq_cols, window=cfg["window_length"])
        frac = float(cfg.get("deep_subsample_frac", 0.6))
        if len(Xs_tr) > 30000:  # stratified-by-engine subsample (>=60%), A4
            euniq = np.unique(us_tr)
            keep_e = np.random.RandomState(SEED).choice(euniq, max(1, int(len(euniq) * frac)), replace=False)
            m = np.isin(us_tr, keep_e)
            Xs_tr, ys_tr, us_tr = Xs_tr[m], ys_tr[m], us_tr[m]
            log.info("%s deep: stratified-by-engine subsample %d/%d engines", fd, len(keep_e), len(euniq))
        nunits_rows = ten.groupby("unit").size().values
        ends = np.cumsum(nunits_rows) - 1
        for kind in ["lstm", "cnn"]:
            pseed, vseed, first = [], [], None
            for s in deep_seeds:
                try:
                    mdl, mu, sd, vrmse = train_torch_model(
                        Xs_tr, ys_tr, us_tr, len(seq_cols), kind=kind,
                        epochs=deep_epochs, patience=deep_patience, seed=s)
                    mdl.eval()
                    with torch.no_grad():
                        Xst = ((Xs_te.reshape(-1, len(seq_cols)) - mu) / sd).reshape(
                            Xs_te.shape).astype(np.float32)
                        pall = mdl(torch.from_numpy(Xst)).numpy()
                    pseed.append(pall[ends])
                    vseed.append(vrmse)
                    if first is None:
                        first = (mdl, mu, sd, pall)
                except Exception as e:
                    log.warning("deep %s seed %d failed on %s: %s", kind, s, fd, e)
            if pseed:
                fitted[kind] = ("torch",) + first[:3] + (seq_cols,)
                seed_test_preds[kind] = pseed
                torch_allrow[kind] = first[3]
                deep_holdout[kind] = vseed
                deep_info[kind] = f"trained seeds={len(pseed)} holdout={np.mean(vseed):.2f}"
            else:
                deep_info[kind] = "failed"

    # ---- test evaluation: CAPPED last-cycle metrics are primary (A8) ----
    from src.evaluate import binned_error_table, bootstrap_ci_by_engine
    reg_rows, bin_rows, preds_last = [], [], {}
    units_last = np.array(sorted(test_all["unit"].unique()))
    for name, arrs in seed_test_preds.items():
        P = np.stack([np.asarray(a, dtype=float) for a in arrs])  # seeds x engines
        p_last = P.mean(0)
        preds_last[name] = p_last
        if name in ("lstm", "cnn"):
            cv = {"cv_rmse_mean": float(np.mean(deep_holdout[name])),
                  "cv_rmse_std": float(np.std(deep_holdout[name])) if len(deep_holdout[name]) > 1 else 0.0,
                  "cv_phm_mean": float("nan"), "cv_phm_std": float("nan"),
                  "cv_note": "by-engine holdout 20%, mean over seeds"}
        else:
            c = cvres.get(name, {})
            cv = {"cv_rmse_mean": c.get("cv_rmse_mean", float("nan")),
                  "cv_rmse_std": c.get("cv_rmse_std", float("nan")),
                  "cv_phm_mean": c.get("cv_phm_mean", float("nan")),
                  "cv_phm_std": c.get("cv_phm_std", float("nan")),
                  "cv_note": f"grouped {n_splits}-fold CV"}
        m_cap = regression_metrics(yte_cap_last, p_last)
        m_true = regression_metrics(yte_true_last, p_last)
        per_seed_cap = [regression_metrics(yte_cap_last, P[i])["rmse"] for i in range(P.shape[0])]
        row = {"fd": fd, "model": name,
               "cv_rmse": cv["cv_rmse_mean"], "cv_rmse_std": cv["cv_rmse_std"],
               "cv_phm": cv["cv_phm_mean"], "cv_phm_std": cv["cv_phm_std"],
               "cv_note": cv["cv_note"],
               "test_last_rmse_cap": m_cap["rmse"], "test_last_mae_cap": m_cap["mae"],
               "test_last_r2_cap": m_cap["r2"], "test_last_phm_cap": m_cap["phm"],
               "test_last_rmse_cap_std": float(np.std(per_seed_cap)) if len(per_seed_cap) > 1 else 0.0,
               "test_last_rmse_true": m_true["rmse"], "test_last_mae_true": m_true["mae"],
               "test_last_r2_true": m_true["r2"], "test_last_phm_true": m_true["phm"]}
        # all-cycle CAPPED metrics for deployed sklearn models (uncapped version dropped: meaningless, A8)
        mdl = fitted.get(name)
        if mdl is not None and not isinstance(mdl, tuple):
            p_all = mdl.predict(Xte)
            mall = regression_metrics(test_all["RUL_cap"].values, p_all)
            row.update({"test_all_rmse_cap": mall["rmse"], "test_all_mae_cap": mall["mae"],
                        "test_all_r2_cap": mall["r2"], "test_all_phm_cap": mall["phm"]})
        else:
            row.update({"test_all_rmse_cap": float("nan"), "test_all_mae_cap": float("nan"),
                        "test_all_r2_cap": float("nan"), "test_all_phm_cap": float("nan")})
        reg_rows.append(row)
        for b in binned_error_table(yte_cap_last, p_last):
            bin_rows.append({"fd": fd, "model": name, **b})
    reg_df = pd.DataFrame(reg_rows)
    pd.DataFrame(bin_rows).to_csv(f"results/{fd}_regression_bins.csv", index=False)

    # bootstrap CIs (resampling engines) for the CV-selected shortlist later;
    # stored per model here for RMSE_cap and PHM_cap
    n_boot = int(cfg.get("bootstrap_resamples", 1000))
    ci_rows = []
    for name in preds_last:
        p_last = preds_last[name]
        for metric_name, fn in [("rmse_cap", lambda yt, yp: regression_metrics(yt, yp)["rmse"]),
                                ("phm_cap", lambda yt, yp: regression_metrics(yt, yp)["phm"])]:
            mean, lo, hi = bootstrap_ci_by_engine(units_last, yte_cap_last, p_last, fn, n_boot, SEED)
            ci_rows.append({"fd": fd, "model": name, "metric": metric_name,
                            "mean": mean, "lo": lo, "hi": hi})
    pd.DataFrame(ci_rows).to_csv(f"results/{fd}_regression_cis.csv", index=False)

    # model selection by CV on capped RMSE (tie-break PHM) — NEVER test (A2)
    sel = reg_df[reg_df["model"] != "mean"].sort_values(["cv_rmse", "cv_phm"])
    best_overall = sel.iloc[0]["model"]
    sk = reg_df[~reg_df["model"].isin(["mean", "lstm", "cnn"])]
    best_name = sk.sort_values(["cv_rmse", "cv_phm"]).iloc[0]["model"]
    log.info("%s CV-best overall=%s deployed-sklearn=%s", fd, best_overall, best_name)
    best_pred_last = preds_last[best_overall]
    best_model = fitted[best_overall]
    deploy_pred_last = preds_last[best_name]
    deploy_model = fitted[best_name]

    # ---- regression plots (CV-best overall model) ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte_cap_last, best_pred_last, alpha=0.6, s=18, color="#0173B2")
    lim = [0, max(yte_cap_last.max(), best_pred_last.max()) * 1.05]
    ax.plot(lim, lim, "k--", label="ideal")
    ax.set_title(f"{fd}: predicted vs true RUL, capped at 125 ({best_overall}, last cycle)")
    ax.set_xlabel("true RUL capped at 125 (cycles)"); ax.set_ylabel("predicted RUL (cycles)")
    ax.legend()
    save(fig, f"figures/{fd}_pred_vs_true_{best_overall}.png")
    # residuals vs capped true
    resid = best_pred_last - yte_cap_last
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(yte_cap_last, resid, alpha=0.6, s=18, color="#CC78BC")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_title(f"{fd}: residuals vs capped true RUL ({best_overall})")
    ax.set_xlabel("true RUL capped at 125 (cycles)"); ax.set_ylabel("residual (pred − true)")
    save(fig, f"figures/{fd}_residuals_{best_overall}.png")
    # per-engine trajectories for 6 engines (best/median/worst + early-failing)
    errs = np.abs(best_pred_last - yte_cap_last)
    order = np.argsort(errs)
    units_sorted = sorted(test_all["unit"].unique())
    picks = set([order[0], order[len(order)//2], order[-1]])
    # early failing: smallest true last RUL
    picks.add(int(np.argmin(yte_true_last)))
    # add 2 more spread
    picks.add(order[len(order)//4]); picks.add(order[3*len(order)//4])
    # full-history predicted RUL per row from the CV-best overall model
    if isinstance(best_model, tuple):
        p_all_best = torch_allrow[best_overall]
    else:
        p_all_best = best_model.predict(Xte)
    if p_all_best is not None and len(p_all_best) == len(test_all):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=False, sharey=True)
        for ax, pi in zip(axes.flat, sorted(picks)[:6]):
            u = units_sorted[pi]
            m = test_all["unit"] == u
            cyc = test_all.loc[m, "cycle"].values
            ax.plot(cyc, test_all.loc[m, "RUL_cap"].values, label="true (capped)", color="k", linewidth=1.5)
            # align predictions: need positional mask
            pm = (test_all["unit"].values == u)
            ax.plot(cyc, p_all_best[pm], label="pred", color="#0173B2", linewidth=1.2)
            ax.set_title(f"unit {u} (err={errs[pi]:.1f})")
            ax.set_xlabel("cycle"); ax.set_ylabel("RUL (capped)")
        axes.flat[0].legend()
        fig.suptitle(f"{fd}: per-engine RUL trajectories ({best_overall})")
        save(fig, f"figures/{fd}_trajectories_pred_{best_overall}.png")

    # ---- classification (A5): CV-selected headlines, tidy long format, no duplicates ----
    y3_last = test_all["cls3_id"].values[last_pos]
    yb_last = test_all["bin30"].values[last_pos]
    y3_tr = train["cls3_id"].values
    yb_tr = train["bin30"].values
    clf_models = get_classifiers(seed=SEED, fast=smoke)
    # grouped 5-fold CV for direct classifiers on both tasks
    cv3 = grouped_cv_classification(Xtr, y3_tr, groups, clf_models, n_splits, SEED)
    cvb = grouped_cv_classification(Xtr, yb_tr, groups, clf_models, n_splits, SEED)
    head3 = max(cv3, key=lambda k: cv3[k]["cv_macro_f1_mean"])
    headb = max(cvb, key=lambda k: cvb[k]["cv_macro_f1_mean"])
    log.info("%s CV-best direct 3-class=%s (macroF1=%.3f) binary=%s (macroF1=%.3f)",
             fd, head3, cv3[head3]["cv_macro_f1_mean"], headb, cvb[headb]["cv_macro_f1_mean"])
    # fit all direct classifiers on full train
    fitted_c3, fitted_cb, proba_cb = {}, {}, {}
    for name, clf in clf_models.items():
        try:
            fitted_c3[name] = clf.fit(Xtr, y3_tr)
        except Exception as e:
            log.warning("clf3 %s %s failed: %s", fd, name, e)
    for name in fitted_c3:
        try:
            cb = get_classifiers(seed=SEED, fast=smoke)[name].fit(Xtr, yb_tr)
            fitted_cb[name] = cb
            try:
                proba_cb[name] = cb.predict_proba(Xte_last)[:, 1]
            except Exception:
                proba_cb[name] = fitted_cb[name].predict(Xte_last).astype(float)
        except Exception as e:
            log.warning("clfb %s %s failed: %s", fd, name, e)
    clf_rows, cm_store = [], {}
    for name in fitted_c3:
        p3 = fitted_c3[name].predict(Xte_last)
        rows3, cm3 = tidy_rows_3class(fd, "direct", name, y3_last, p3)
        clf_rows.extend(rows3)
        cm_store[("3class", name)] = (cm3, p3)
        if name in fitted_cb:
            pb = fitted_cb[name].predict(Xte_last)
            rowsb, cmb = tidy_rows_binary(fd, "direct", name, yb_last, pb, proba_cb[name])
            clf_rows.extend(rowsb)
            cm_store[("binary30", name)] = (cmb, pb)
    # regress-then-threshold computed ONCE with the deployed (CV-best sklearn) regressor
    r3, rb = rul_to_classes(deploy_pred_last)
    rows3r, cm3r = tidy_rows_3class(fd, "regress-threshold", f"{best_name}-via-RUL", y3_last, r3)
    clf_rows.extend(rows3r)
    cm_store[("3class", f"{best_name}-via-RUL")] = (cm3r, r3)
    rowsbr, cmbr = tidy_rows_binary(fd, "regress-threshold", f"{best_name}-via-RUL",
                                    yb_last, rb, -deploy_pred_last)
    clf_rows.extend(rowsbr)
    cm_store[("binary30", f"{best_name}-via-RUL")] = (cmbr, rb)
    clf_df = pd.DataFrame(clf_rows)
    assert not clf_df.duplicated(subset=["fd", "task", "approach", "model", "class"]).any(), \
        "duplicate rows in classification CSV"
    clf_df.to_csv(f"results/{fd}_classification.csv", index=False)
    # figures for the CV-best headlines (counts + row-normalised, labelled axes)
    for (task, hname, labels) in [("3class", head3, CLASS3_LABELS),
                                  ("binary30", headb, ["neg(≥31)", "pos(≤30)"])]:
        cm, _ = cm_store[(task, hname)]
        tag = "3class" if task == "3class" else "binary"
        plot_labeled_cm(cm, labels, f"{fd}: {tag} confusion matrix ({hname} direct, last cycle)",
                        f"figures/{fd}_cm_{tag}_{hname}.png")
        plot_labeled_cm(cm, labels, f"{fd}: {tag} row-normalised ({hname} direct)",
                        f"figures/{fd}_cm_{tag}_{hname}_norm.png", normalize=True)
    # ROC/PR for headline binary
    if len(np.unique(yb_last)) > 1:
        from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
        sc = proba_cb[headb]
        auc_txt = clf_df[(clf_df.task == "binary30") & (clf_df.approach == "direct")
                         & (clf_df.model == headb) & (clf_df["class"] == "macro")]["roc_auc"].values[0]
        ap_txt = clf_df[(clf_df.task == "binary30") & (clf_df.approach == "direct")
                        & (clf_df.model == headb) & (clf_df["class"] == "macro")]["pr_auc"].values[0]
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(yb_last, sc, ax=ax)
        ax.set_title(f"{fd}: ROC (binary ≤30, {headb}, AUC={auc_txt:.3f})")
        save(fig, f"figures/{fd}_roc_{headb}.png")
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(yb_last, sc, ax=ax)
        ax.set_title(f"{fd}: PR curve (AP={ap_txt:.3f}, positives rare)")
        save(fig, f"figures/{fd}_pr_{headb}.png")
    # bootstrap CIs: critical recall + macro-F1 (3-class), positive recall + F1 (binary)
    from sklearn.metrics import f1_score, recall_score
    from src.evaluate import bootstrap_ci_by_engine as _bci
    ci_rows = []
    _cm3, _p3h = cm_store[("3class", head3)]
    _cmb, _pbh = cm_store[("binary30", headb)]
    _cm3r, _p3r = cm_store[("3class", f"{best_name}-via-RUL")]
    _cmbr, _pbr = cm_store[("binary30", f"{best_name}-via-RUL")]
    ci_specs = [
        ("3class", "direct", head3, y3_last, _p3h,
         lambda a, b: recall_score(a, b, labels=[0, 1, 2], average=None, zero_division=0)[2],
         "critical_recall"),
        ("3class", "direct", head3, y3_last, _p3h,
         lambda a, b: f1_score(a, b, average="macro", zero_division=0), "macro_f1"),
        ("3class", "regress-threshold", f"{best_name}-via-RUL", y3_last, _p3r,
         lambda a, b: recall_score(a, b, labels=[0, 1, 2], average=None, zero_division=0)[2],
         "critical_recall"),
        ("3class", "regress-threshold", f"{best_name}-via-RUL", y3_last, _p3r,
         lambda a, b: f1_score(a, b, average="macro", zero_division=0), "macro_f1"),
        ("binary30", "direct", headb, yb_last, _pbh,
         lambda a, b: recall_score(a, b, pos_label=1, zero_division=0), "positive_recall"),
        ("binary30", "direct", headb, yb_last, _pbh,
         lambda a, b: f1_score(a, b, pos_label=1, zero_division=0), "positive_f1"),
        ("binary30", "regress-threshold", f"{best_name}-via-RUL", yb_last, _pbr,
         lambda a, b: recall_score(a, b, pos_label=1, zero_division=0), "positive_recall"),
        ("binary30", "regress-threshold", f"{best_name}-via-RUL", yb_last, _pbr,
         lambda a, b: f1_score(a, b, pos_label=1, zero_division=0), "positive_f1"),
    ]
    for task, appr, mname, yt, yp, fn, metric in ci_specs:
        mean, lo, hi = _bci(units_last, yt, yp, fn, n_boot, SEED)
        ci_rows.append({"fd": fd, "task": task, "approach": appr, "model": mname,
                        "metric": metric, "mean": mean, "lo": lo, "hi": hi})
    pd.DataFrame(ci_rows).to_csv(f"results/{fd}_classification_cis.csv", index=False)

    # ---- threshold tuning on OOF predictions of the DEPLOYED model (A6) ----
    from sklearn.base import clone
    from sklearn.metrics import (confusion_matrix as _cm, precision_score as _ps,
                                 recall_score as _rs)
    from src.evaluate import expected_cost as _ecost
    from src.train import oof_predict as _oof
    miss_cost = float(cfg.get("cost_matrix", {}).get("missed_failure", 10.0))
    fa_cost = float(cfg.get("cost_matrix", {}).get("false_alarm", 1.0))

    def _deploy_factory():
        if best_name == "hgb_tuned":
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(**best_params, random_state=SEED)
        return get_regressors(seed=SEED, fast=smoke)[best_name]

    try:
        oof = _oof(_deploy_factory, Xtr, ytr_cap, groups, n_splits, SEED)
        # realistic operating points: random truncation cuts per train engine
        rng = np.random.RandomState(SEED)
        tru_units = train["unit"].values
        tru_rul = train["RUL"].values
        pv_list, yt_list = [], []
        for _ in range(3):
            for u in np.unique(tru_units):
                idx = np.where(tru_units == u)[0]
                cut = rng.choice(idx)
                pv_list.append(oof[cut]); yt_list.append(float(tru_rul[cut]))
        pv = np.array(pv_list); yt_tr = np.array(yt_list)
        yv = (yt_tr <= 30).astype(int)
        ts = np.arange(10, 61, 2)
        curve = []
        for t in ts:
            pp = (pv <= t).astype(int)
            cm = _cm(yv, pp, labels=[0, 1])
            fn, fp = int(cm[1, 0]), int(cm[0, 1])
            curve.append({"t": float(t),
                          "recall": float(_rs(yv, pp, zero_division=0)),
                          "precision": float(_ps(yv, pp, zero_division=0)),
                          "cost10": float(fn * miss_cost + fp * fa_cost)})
        feas = [c for c in curve if c["recall"] >= 0.95]
        if feas:
            pick = min(feas, key=lambda c: (c["cost10"], -c["precision"]))
            rule_used, target_met = "min cost subject to OOF recall>=0.95", True
        else:
            pick = min(curve, key=lambda c: c["cost10"])
            rule_used, target_met = f"min expected cost ({miss_cost:g}:{fa_cost:g}); 0.95 recall infeasible on OOF", False
        best_t = float(pick["t"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([c["recall"] for c in curve], [c["precision"] for c in curve], marker="o")
        ax.set_title(f"{fd}: OOF PR trade-off at truncation points ({best_name}; frozen for test)")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        save(fig, f"figures/{fd}_thresh_tuning.png")
        # frozen threshold -> test (deployed sklearn model)
        pb_tuned = (deploy_pred_last <= best_t).astype(int)
        mb_tuned = binary_metrics(yb_last, pb_tuned, (-deploy_pred_last))
        pb_default = (deploy_pred_last <= 30).astype(int)
        cost_default = _ecost(binary_metrics(yb_last, pb_default)["confusion_matrix"],
                              miss_cost, fa_cost)
        cost_tuned = _ecost(mb_tuned["confusion_matrix"], miss_cost, fa_cost)
        t_rec = float(_rs(yb_last, pb_tuned, zero_division=0))
        t_pre = float(_ps(yb_last, pb_tuned, zero_division=0))
        _, t_rec_lo, t_rec_hi = _bci(units_last, yb_last, pb_tuned,
                                     lambda a, b: _rs(a, b, zero_division=0), n_boot, SEED)
        _, t_pre_lo, t_pre_hi = _bci(units_last, yb_last, pb_tuned,
                                     lambda a, b: _ps(a, b, zero_division=0), n_boot, SEED)
        # cost-ratio sensitivity on OOF + frozen-to-test cost
        sens_rows = []
        for ratio in [5, 10, 20, 50]:
            scored = []
            for c in curve:
                pp = (pv <= c["t"]).astype(int)
                cm = _cm(yv, pp, labels=[0, 1])
                scored.append((c["t"], float(int(cm[1, 0]) * ratio + int(cm[0, 1]))))
            t_opt = min(scored, key=lambda x: x[1])[0]
            pb_s = (deploy_pred_last <= t_opt).astype(int)
            cm_s = _cm(yb_last, pb_s, labels=[0, 1])
            sens_rows.append({"fd": fd, "fn_fp_ratio": ratio, "t_opt_oof": float(t_opt),
                             "test_cost": float(int(cm_s[1, 0]) * ratio + int(cm_s[0, 1])),
                             "test_recall": float(_rs(yb_last, pb_s, zero_division=0)),
                             "test_precision": float(_ps(yb_last, pb_s, zero_division=0))})
        pd.DataFrame(sens_rows).to_csv(f"results/{fd}_cost_sensitivity.csv", index=False)
    except Exception as e:
        log.warning("threshold tuning failed %s: %s", fd, e)
        best_t, rule_used, target_met = 30.0, "fallback (tuning failed)", False
        t_rec = t_pre = float("nan"); t_rec_lo = t_rec_hi = t_pre_lo = t_pre_hi = float("nan")
        cost_default = cost_tuned = float("nan")

    # ---- lead-time, premature alerts, any-time detection (deployed sklearn model) ----
    p_all = deploy_model.predict(Xte)
    lt = lead_time_analysis(test_all.reset_index(drop=True), p_all)
    lt.to_csv(f"results/{fd}_leadtime_{best_name}.csv", index=False)
    far = false_alarm_rate(test_all, p_all)
    wl = lt["warn_lead"].dropna().values; cl = lt["crit_lead"].dropna().values
    lead = {"warn_median": float(np.median(wl)) if len(wl) else float("nan"),
            "warn_iqr": [float(np.percentile(wl, 25)), float(np.percentile(wl, 75))] if len(wl) else [float("nan"), float("nan")],
            "crit_median": float(np.median(cl)) if len(cl) else float("nan"),
            "crit_iqr": [float(np.percentile(cl, 25)), float(np.percentile(cl, 75))] if len(cl) else [float("nan"), float("nan")],
            "premature_warn_frac": float(np.mean(wl > 60)) if len(wl) else float("nan"),
            "premature_crit_frac": float(np.mean(cl > 30)) if len(cl) else float("nan")}
    # any-time critical detection: true last RUL<20 engines ever flagged critical
    te_units = test_all["unit"].values
    det, tot = 0, 0
    for u in np.unique(te_units):
        mrow = te_units == u
        if float(test_all.loc[mrow, "RUL_last"].iloc[0]) < 20:
            tot += 1
            if bool((p_all[mrow] < 20).any()):
                det += 1
    lead["anytime_crit_rate"] = float(det / tot) if tot else float("nan")
    lead["anytime_crit_n"] = [int(det), int(tot)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cl, bins=20, color="#CC78BC", label="critical lead")
    ax.hist(wl, bins=20, color="#DE8F05", alpha=0.6, label="warning lead")
    ax.set_title(f"{fd}: alert lead-time distribution ({best_name})")
    ax.set_xlabel("cycles before true failure when first alert raised")
    ax.set_ylabel("engines"); ax.legend()
    save(fig, f"figures/{fd}_leadtime_{best_name}.png")

    # ---- save artefacts ----
    reg_df.to_csv(f"results/{fd}_regression.csv", index=False)
    with open(f"results/{fd}_summary.json", "w") as f:
        json.dump({"summary": summ, "cv": cvres, "best_hgb_params": best_params,
                   "best_overall": best_overall, "best_model": best_name,
                   "sensors_kept": store["sensors"], "sensors_dropped": store["dropped"],
                   "feat_cols": feat_cols,
                   "threshold": {"t": float(best_t), "rule_used": rule_used,
                                 "target_met": bool(target_met),
                                 "test_recall": t_rec, "test_recall_ci": [t_rec_lo, t_rec_hi],
                                 "test_precision": t_pre, "test_precision_ci": [t_pre_lo, t_pre_hi]},
                   "lead_time": lead,
                   "false_alarm_per_100": float(far),
                   "cost_default": float(cost_default) if cost_default == cost_default else None,
                   "cost_tuned": float(cost_tuned) if cost_tuned == cost_tuned else None,
                   "deep": deep_info}, f, indent=2, default=str)
    # persist deployed sklearn model (app-compatible) + torch bundle if overall-best is deep
    with open(f"models/{fd}_{best_name}.pkl", "wb") as f:
        pickle.dump({"model": deploy_model, "feat_cols": feat_cols, "sensors": store["sensors"],
                     "scaler": store["scaler"],
                     "scaler_mean": store["scaler"].mean_.tolist(),
                     "scaler_scale": store["scaler"].scale_.tolist(),
                     "norm": store["norm"], "windows": cfg["rolling_windows"],
                     "smooth_alpha": cfg.get("smooth_alpha", 0.3),
                     "mode": ("cluster" if fd in ("FD002", "FD004") else "global")}, f)
    if isinstance(best_model, tuple):
        import torch as _tsave
        _, tm, tmu, tsd, tcols = best_model
        _tsave.save({"state_dict": tm.state_dict(), "mu": tmu, "sd": tsd, "sensors": tcols,
                     "window": cfg["window_length"]}, f"models/{fd}_{best_overall}_torch.pt")
    # ---- importance on a grouped holdout, capped targets, neg-RMSE scoring (A7) ----
    from sklearn.model_selection import GroupShuffleSplit
    from src.explain import (grouped_permutation_importance, permutation_top,
                             shap_beeswarm)
    try:
        gss = GroupShuffleSplit(1, test_size=0.25, random_state=SEED)
        hi_tr, hi_va = next(gss.split(Xtr, ytr_cap, groups))
        imp_model = clone(deploy_model).fit(Xtr[hi_tr], ytr_cap[hi_tr])
        Xa_va, ya_va = Xtr[hi_va], ytr_cap[hi_va]
        gimp = grouped_permutation_importance(imp_model, Xa_va, ya_va, feat_cols,
                                              n_repeats=10, seed=SEED)
        gimp.to_csv(f"results/{fd}_importance_sensor_{best_name}.csv", index=False)
        fimp = permutation_top(imp_model, Xa_va, ya_va, feat_cols, n_repeats=10,
                               seed=SEED, top_k=15, scoring="neg_root_mean_squared_error")
        fimp.to_csv(f"results/{fd}_permutation_{best_name}.csv", index=False)
        top5 = gimp["sensor"].head(5).tolist()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(gimp["sensor"][::-1].values, gimp["importance"][::-1].values, color="#0173B2")
        ax.set_title(f"{fd}: per-sensor importance ({best_name}; top: {', '.join(top5)})")
        ax.set_xlabel("mean neg-RMSE drop shuffling sensor group (capped RUL, holdout)")
        save(fig, f"figures/{fd}_importance_sensor_{best_name}.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fimp["feature"][::-1].values, fimp["importance"][::-1].values, color="#029E73")
        ax.set_title(f"{fd}: per-feature importance ({best_name}, top-15)")
        ax.set_xlabel("mean neg-RMSE drop (capped RUL, holdout)")
        save(fig, f"figures/{fd}_permutation_{best_name}.png")
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor, RandomForestClassifier,
                                      RandomForestRegressor)
        shap_ok = isinstance(imp_model, (HistGradientBoostingRegressor, RandomForestRegressor,
                                        HistGradientBoostingClassifier, RandomForestClassifier)) \
            and shap_beeswarm(imp_model, Xa_va, feat_cols,
                              f"figures/{fd}_shap_{best_name}.png")
        log.info("%s importance done (SHAP=%s)", fd, shap_ok)
    except Exception as e:
        log.warning("importance failed %s: %s", fd, e)
    log.info("%s done. overall=%s deployed=%s", fd, best_overall, best_name)
    return {"fd": fd, "reg": reg_df, "clf": clf_df, "summary": summ,
            "best": best_overall, "deployed": best_name}


def _last_pos_index(test_all):
    last_idx = test_all.groupby("unit")["cycle"].idxmax().sort_values()
    pos = {ix: i for i, ix in enumerate(test_all.index)}
    return np.array([pos[i] for i in last_idx.values])


def run_ablations(cfg, subsets=("FD001", "FD002")):
    """Ablation study with explicit columns (A10).

    Columns: subset, ablation, setting, metric_name, cv_mean, cv_std, test_value.
    Each ablation trains default HGB (seed 42) with grouped 5-fold CV on capped
    RUL, plus a frozen test-last-cycle value. Cap ablation compares on a COMMON
    target (predictions and truth clipped to 125) so caps are comparable.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold
    from src.train import prepare_subset, build_feature_store
    from src.evaluate import regression_metrics
    rows = []

    def _eval(fd, store, train, test_all, clf_target=None):
        """Returns (cv_mean, cv_std, test_value) for regression or macro-F1."""
        Xtr, Xte = store["Xtr"], store["Xte"]
        groups = train["unit"].values
        lp = _last_pos_index(test_all)
        gkf = GroupKFold(5)
        if clf_target is not None:
            y = train[clf_target].values
            cvs = []
            for tr, va in gkf.split(Xtr, y, groups):
                m = HistGradientBoostingClassifier(random_state=SEED,
                                                   class_weight="balanced").fit(Xtr[tr], y[tr])
                cvs.append(f1_score(y[va], m.predict(Xtr[va]), average="macro", zero_division=0))
            m = HistGradientBoostingClassifier(random_state=SEED,
                                               class_weight="balanced").fit(Xtr, y)
            te = f1_score(test_all[clf_target].values[lp], m.predict(Xte[lp]),
                          average="macro", zero_division=0)
            return float(np.mean(cvs)), float(np.std(cvs)), float(te)
        y = train["RUL_cap"].values
        cvs = []
        for tr, va in gkf.split(Xtr, y, groups):
            m = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr[tr], y[tr])
            cvs.append(np.sqrt(np.mean((y[va] - m.predict(Xtr[va])) ** 2)))
        m = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr, y)
        te = float(np.sqrt(np.mean(
            (test_all["RUL_cap"].values[lp] - m.predict(Xte[lp])) ** 2)))
        return float(np.mean(cvs)), float(np.std(cvs)), te

    for fd in subsets:
        train, test_all, _ = prepare_subset(fd, "data/raw", cfg)
        base = build_feature_store(train.copy(), test_all.copy(), cfg, fd)
        cm, cs, te = _eval(fd, base, train, test_all)
        b_cv, b_sd, b_te = cm, cs, te
        rows.append({"subset": fd, "ablation": "baseline", "setting": "default",
                     "metric_name": "rmse_cap", "cv_mean": cm, "cv_std": cs, "test_value": te})
        # smooth order: correct (norm->smooth) vs legacy (smooth->raw->norm)
        from src.preprocess import (ConditionNormalizer, feature_matrix, smooth_per_unit,
                                    select_sensors_by_variance)
        from sklearn.preprocessing import StandardScaler
        from src.data import SENSORS as _S
        train_l, test_l = train.copy(), test_all.copy()
        keep0, _ = select_sensors_by_variance(train_l, list(_S), cfg.get("variance_threshold", 1e-6))
        a = cfg.get("smooth_alpha", 0.3)
        tsl = smooth_per_unit(train_l, keep0, a); tsl2 = smooth_per_unit(test_l, keep0, a)
        for c in keep0:
            train_l[c] = tsl[c + "_sm"]; test_l[c] = tsl2[c + "_sm"]
        mode = "cluster" if fd in ("FD002", "FD004") else "global"
        nl = ConditionNormalizer(mode=mode, k=cfg.get("kmeans_k", 6),
                                 random_state=cfg.get("seed", 42)).fit(train_l, keep0)
        Xtr_l, fc = feature_matrix(nl.transform(train_l), keep0, cfg["rolling_windows"])
        Xte_l, _ = feature_matrix(nl.transform(test_l), keep0, cfg["rolling_windows"])
        sc = StandardScaler().fit(Xtr_l.values)
        leg = {"Xtr": sc.transform(Xtr_l.values), "Xte": sc.transform(Xte_l.values)}
        cm, cs, te = _eval(fd, leg, train, test_all)
        rows.append({"subset": fd, "ablation": "smooth_order", "setting": "legacy_smooth_first",
                     "metric_name": "rmse_cap", "cv_mean": cm, "cv_std": cs, "test_value": te})
        rows.append({"subset": fd, "ablation": "smooth_order", "setting": "fixed_norm_first",
                     "metric_name": "rmse_cap", "cv_mean": b_cv,
                     "cv_std": b_sd, "test_value": b_te})
        # condition normalisation on/off (multi-condition subsets)
        if fd in ("FD002", "FD004"):
            off = build_feature_store(train.copy(), test_all.copy(), cfg, fd,
                                      norm_mode_override="global")
            cm, cs, te = _eval(fd, off, train, test_all)
            rows.append({"subset": fd, "ablation": "condnorm", "setting": "off_global",
                         "metric_name": "rmse_cap", "cv_mean": cm, "cv_std": cs, "test_value": te})
            rows.append({"subset": fd, "ablation": "condnorm", "setting": "on_cluster",
                         "metric_name": "rmse_cap", "cv_mean": b_cv,
                         "cv_std": b_sd, "test_value": b_te})
        # rolling windows
        for w in [[10], [10, 30], [5, 10, 30]]:
            st = build_feature_store(train.copy(), test_all.copy(), cfg, fd, windows_override=w)
            cm, cs, te = _eval(fd, st, train, test_all)
            rows.append({"subset": fd, "ablation": "rolling", "setting": str(w),
                         "metric_name": "rmse_cap", "cv_mean": cm, "cv_std": cs, "test_value": te})
        # RUL cap on a COMMON target (clip preds+truth to 125)
        for cap in cfg.get("rul_cap_ablation", [100, 125, 130]):
            tr2, te2, _ = prepare_subset(fd, "data/raw", cfg, rul_cap=cap)
            st = build_feature_store(tr2.copy(), te2.copy(), cfg, fd)
            Xtr2, Xte2 = st["Xtr"], st["Xte"]
            g2 = tr2["unit"].values
            y2 = tr2["RUL_cap"].values
            lp2 = _last_pos_index(te2)
            cvs = []
            for tr, va in GroupKFold(5).split(Xtr2, y2, g2):
                m = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr2[tr], y2[tr])
                pc = np.minimum(m.predict(Xtr2[va]), 125)
                yc = np.minimum(y2[va], 125)
                cvs.append(float(np.sqrt(np.mean((yc - pc) ** 2))))
            m = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr2, y2)
            pte = np.minimum(m.predict(Xte2[lp2]), 125)
            yte = np.minimum(te2["RUL"].values[lp2], 125)
            te = float(np.sqrt(np.mean((yte - pte) ** 2)))
            rows.append({"subset": fd, "ablation": "rul_cap", "setting": str(cap),
                         "metric_name": "rmse_common125", "cv_mean": float(np.mean(cvs)),
                         "cv_std": float(np.std(cvs)), "test_value": te})
        # with/without cycle and with/without op settings (drop features post-hoc)
        for ab, dropcols in [("no_cycle", ["cycle"]), ("no_ops", ["op1", "op2", "op3"])]:
            keep_idx = [i for i, c in enumerate(base["feat_cols"]) if c not in dropcols]
            Xtr_d = base["Xtr"][:, keep_idx]; Xte_d = base["Xte"][:, keep_idx]
            y = train["RUL_cap"].values
            groups = train["unit"].values
            lp = _last_pos_index(test_all)
            cvs = []
            for tr, va in GroupKFold(5).split(Xtr_d, y, groups):
                m = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr_d[tr], y[tr])
                cvs.append(float(np.sqrt(np.mean((y[va] - m.predict(Xtr_d[va])) ** 2))))
            m = HistGradientBoostingRegressor(random_state=SEED).fit(Xtr_d, y)
            te = float(np.sqrt(np.mean(
                (test_all["RUL_cap"].values[lp] - m.predict(Xte_d[lp])) ** 2)))
            rows.append({"subset": fd, "ablation": ab, "setting": "dropped",
                         "metric_name": "rmse_cap", "cv_mean": float(np.mean(cvs)),
                         "cv_std": float(np.std(cvs)), "test_value": te})
        # class-weight on/off (3-class macro-F1)
        cw_bal = None
        for cw in ["balanced", None]:
            from sklearn.ensemble import HistGradientBoostingClassifier as _HGBC
            Xtr_c, Xte_c = base["Xtr"], base["Xte"]
            y = train["cls3_id"].values
            groups = train["unit"].values
            lp = _last_pos_index(test_all)
            cvs = []
            for tr, va in GroupKFold(5).split(Xtr_c, y, groups):
                m = _HGBC(random_state=SEED, class_weight=cw).fit(Xtr_c[tr], y[tr])
                cvs.append(f1_score(y[va], m.predict(Xtr_c[va]), average="macro", zero_division=0))
            m = _HGBC(random_state=SEED, class_weight=cw).fit(Xtr_c, y)
            te = f1_score(test_all["cls3_id"].values[lp], m.predict(Xte_c[lp]),
                          average="macro", zero_division=0)
            if cw == "balanced":
                cw_bal = (float(np.mean(cvs)), float(np.std(cvs)), float(te))
            rows.append({"subset": fd, "ablation": "class_weight", "setting": str(cw),
                         "metric_name": "macro_f1", "cv_mean": float(np.mean(cvs)),
                         "cv_std": float(np.std(cvs)), "test_value": float(te)})
        # direct vs regress-then-threshold (macro-F1, test + OOF-CV via deployed HGB)
        from src.train import oof_predict as _oof2
        from src.train import rul_to_classes as _r2c
        oof = _oof2(lambda: HistGradientBoostingRegressor(random_state=SEED),
                    base["Xtr"], train["RUL_cap"].values, train["unit"].values, 5, SEED)
        cvs = []
        for tr, va in GroupKFold(5).split(base["Xtr"], y, train["unit"].values):
            c3v, _ = _r2c(oof[va])
            cvs.append(f1_score(train["cls3_id"].values[va], c3v, average="macro", zero_division=0))
        mfull = HistGradientBoostingRegressor(random_state=SEED).fit(base["Xtr"], train["RUL_cap"].values)
        c3t, _ = _r2c(mfull.predict(base["Xte"][_last_pos_index(test_all)]))
        te = f1_score(test_all["cls3_id"].values[_last_pos_index(test_all)], c3t,
                      average="macro", zero_division=0)
        rows.append({"subset": fd, "ablation": "r_vs_c", "setting": "regress_then_threshold",
                     "metric_name": "macro_f1", "cv_mean": float(np.mean(cvs)),
                     "cv_std": float(np.std(cvs)), "test_value": float(te)})
        rows.append({"subset": fd, "ablation": "r_vs_c", "setting": "direct_classifier",
                     "metric_name": "macro_f1", "cv_mean": cw_bal[0],
                     "cv_std": cw_bal[1], "test_value": cw_bal[2]})
    df = pd.DataFrame(rows)
    df.to_csv("results/ablations.csv", index=False)
    # figures: cap + rolling per subset
    for fd in subsets:
        sub = df[(df.subset == fd) & (df.ablation == "rul_cap")]
        if len(sub):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(sub["setting"].astype(str), sub["test_value"], color="#0173B2")
            ax.set_title(f"{fd} ablation: RUL cap vs test RMSE on common capped-125 target")
            ax.set_xlabel("cap"); ax.set_ylabel("RMSE (common-125 target)")
            save(fig, f"figures/{fd}_ablation_cap.png")
    return df


def conformal_interval(fd, cfg, deployed_name=None, best_params=None, smoke=False):
    """Mondrian (predicted-RUL-bin) split conformal on the DEPLOYED model (A9).

    Calibration residuals come from out-of-fold predictions at random
    truncation points (same operating-point scheme as threshold tuning).
    Bins are defined on PREDICTED RUL (observable at deployment):
    [0,20) / [20,50) / [50,125]. Results overwritten (mode "w") every run.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src.train import prepare_subset, build_feature_store, oof_predict
    train, test_all, _ = prepare_subset(fd, "data/raw", cfg)
    store = build_feature_store(train.copy(), test_all.copy(), cfg, fd)
    Xtr, Xte = store["Xtr"], store["Xte"]
    ytr = train["RUL_cap"].values
    groups = train["unit"].values
    n_splits = int(cfg.get("n_splits_cv", 5))
    if deployed_name == "hgb_tuned" and best_params:
        fac = lambda: HistGradientBoostingRegressor(**best_params, random_state=42)  # noqa: E731
    elif deployed_name in ("ridge", "rf", "hgb", "mlp"):
        from src.models import get_regressors
        fac = lambda: get_regressors(seed=42, fast=smoke)[deployed_name]  # noqa: E731
    else:
        fac = lambda: HistGradientBoostingRegressor(random_state=42)  # noqa: E731
    oof = oof_predict(fac, Xtr, ytr, groups, n_splits, 42)
    # calibration pairs at truncation points
    rng = np.random.RandomState(42)
    tru_units = train["unit"].values
    tru_rul = train["RUL"].values
    cpred, cres = [], []
    for _ in range(3):
        for u in np.unique(tru_units):
            idx = np.where(tru_units == u)[0]
            cut = rng.choice(idx)
            cpred.append(oof[cut])
            cres.append(abs(float(tru_rul[cut]) - oof[cut]))
    cpred = np.array(cpred)
    cresid = np.array(cres)
    bins = [(0, 20), (20, 50), (50, 125)]
    quantiles = {}
    for lo, hi in bins:
        m = (cpred >= lo) & (cpred < hi if hi < 125 else cpred <= hi)
        r = cresid[m] if m.sum() >= 20 else cresid  # fall back to global if bin too small
        quantiles[f"{lo}-{hi}"] = {tag: float(np.quantile(r, 1 - a))
                                   for a, tag in [(0.1, "90"), (0.05, "95")]}
    # refit deployed model on full train; evaluate on test last-cycle
    mdl = fac().fit(Xtr, ytr)
    lp = _last_pos_index(test_all)
    p = mdl.predict(Xte[lp])
    yt = test_all["RUL_cap"].values[lp]
    out_lines, records = [], []
    for tag, alpha in [("90", 0.1), ("95", 0.05)]:
        # per-engine half-width from its predicted-RUL bin
        hw = np.zeros(len(p))
        for lo, hi in bins:
            m = (p >= lo) & (p < hi if hi < 125 else p <= hi)
            hw[m] = quantiles[f"{lo}-{hi}"][tag]
        covered = np.abs(yt - p) <= hw
        cov = float(covered.mean())
        se = float(np.sqrt(cov * (1 - cov) / len(p)))
        out_lines.append(
            f"{tag}% Mondrian interval: mean half-width={hw.mean():.2f} "
            f"empirical coverage (capped RUL, last cycle)={cov:.3f} ± {se:.3f} (binomial SE, n={len(p)})")
        for (lo, hi) in bins:
            m = (yt >= lo) & (yt <= hi)
            if m.sum():
                covb = float(covered[m].mean())
                seb = float(np.sqrt(covb * (1 - covb) / m.sum()))
                records.append({"fd": fd, "level": tag, "true_rul_bin": f"{lo}-{hi}",
                                "half_width": float(hw[m].mean()), "coverage": covb,
                                "se": seb, "n": int(m.sum())})
    with open(f"results/{fd}_conformal.txt", "w") as f:
        f.write("\n".join(out_lines) + "\n")
    pd.DataFrame(records).to_csv(f"results/{fd}_conformal_bins.csv", index=False)
    log.info("%s conformal: %s", fd, out_lines)
    return {"quantiles": quantiles, "lines": out_lines}


def deep_window_ablation(cfg, subsets=("FD001", "FD002"), smoke=False):
    """Window-length ablation (20/30/50) for LSTM + CNN, single seed (A4)."""
    import torch
    from src.preprocess import make_sequences as _ms
    from src.train import prepare_subset, build_feature_store
    rows = []
    for fd in subsets:
        train, test_all, _ = prepare_subset(fd, "data/raw", cfg)
        store = build_feature_store(train.copy(), test_all.copy(), cfg, fd)
        seq_cols = store["sensors"]
        trn, ten = store["train"], store["test"]
        lp = _last_pos_index(test_all)
        yt = test_all["RUL_cap"].values[lp]
        nunits_rows = ten.groupby("unit").size().values
        ends = np.cumsum(nunits_rows) - 1
        for w in cfg.get("window_ablation", [20, 30, 50]):
            Xs_tr, ys_tr, us_tr = _ms(trn, seq_cols, window=w)
            Xs_te, _, _ = _ms(ten, seq_cols, window=w)
            for kind in ["lstm", "cnn"]:
                try:
                    mdl, mu, sd, vrmse = train_torch_model(
                        Xs_tr, ys_tr, us_tr, len(seq_cols), kind=kind,
                        epochs=3 if smoke else int(cfg.get("deep_epochs", 60)),
                        patience=int(cfg.get("deep_patience", 8)), seed=42)
                    mdl.eval()
                    with torch.no_grad():
                        Xst = ((Xs_te.reshape(-1, len(seq_cols)) - mu) / sd).reshape(
                            Xs_te.shape).astype(np.float32)
                        p = mdl(torch.from_numpy(Xst)).numpy()[ends]
                    te = float(np.sqrt(np.mean((yt - p) ** 2)))
                    rows.append({"subset": fd, "model": kind, "window": w,
                                 "holdout_rmse": vrmse, "test_rmse_cap": te})
                except Exception as e:
                    log.warning("window ablation %s %s w=%d failed: %s", fd, kind, w, e)
    df = pd.DataFrame(rows)
    df.to_csv("results/deep_window_ablation.csv", index=False)
    return df


def main():
    import sys
    import time
    fresh = "--fresh" in sys.argv
    smoke = "--smoke" in sys.argv
    only = [a.split("=")[1] for a in sys.argv if a.startswith("--only=")]
    only = only[0].split(",") if only else None
    ensure_dirs()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    all_reg, all_clf = [], []
    subsets = [s for s in SUBSETS if (only is None or s in only)]
    wall = {}
    for fd in subsets:
        if not fresh and (Path(f"results/{fd}_regression.csv").exists()
                          and Path(f"results/{fd}_classification.csv").exists()):
            log.warning("%s: REUSING existing results from disk "
                        "(run with --fresh to recompute everything)", fd)
            all_reg.append(pd.read_csv(f"results/{fd}_regression.csv"))
            all_clf.append(pd.read_csv(f"results/{fd}_classification.csv"))
            continue
        t0 = time.time()
        # deep models on all subsets; full configs unless --smoke
        r = run_subset(fd, cfg, do_deep=True,
                       n_iter=int(cfg.get("randomized_search_iter", 12)), smoke=smoke)
        wall[fd] = time.time() - t0
        log.info("%s wall-clock: %.0fs", fd, wall[fd])
        all_reg.append(r["reg"]); all_clf.append(r["clf"])
        try:
            summ_bp = json.load(open(f"results/{fd}_summary.json")).get("best_hgb_params", {})
            ci = conformal_interval(fd, cfg, deployed_name=r.get("deployed"),
                                    best_params=summ_bp, smoke=smoke)
            log.info("%s conformal stored", fd)
        except Exception as e:
            log.warning("conformal %s failed: %s", fd, e)
    if len(subsets) == len(SUBSETS):
        pd.concat(all_reg).to_csv("results/master_regression.csv", index=False)
        pd.concat(all_clf).to_csv("results/master_classification.csv", index=False)
    else:
        log.warning("--only run: master CSVs NOT overwritten (partial subset set)")
    # test-set class counts per subset (A10)
    from src.train import prepare_subset as _prep
    tc_rows = []
    for fd in SUBSETS:
        _, test_all, _ = _prep(fd, "data/raw", cfg)
        c = test_all["cls3"].value_counts()
        tc_rows.append({"fd": fd, "n_engines": int(test_all["unit"].nunique()),
                        "safe": int((test_all["cls3"] == "safe").sum()),
                        "warning": int((test_all["cls3"] == "warning").sum()),
                        "critical": int((test_all["cls3"] == "critical").sum()),
                        "leq30": int((test_all["bin30"] == 1).sum()),
                        "note": "counts over ALL test rows; last-cycle counts differ"})
    pd.DataFrame(tc_rows).to_csv("results/test_class_counts.csv", index=False)
    if not smoke and only is None:
        run_ablations(cfg, subsets=("FD001", "FD002"))
        deep_window_ablation(cfg, subsets=("FD001", "FD002"), smoke=False)
    else:
        log.warning("--smoke/--only: skipping ablations and deep window study")
    # master comparison heatmap + bars on the CAPPED metric (A8)
    if len(subsets) == len(SUBSETS):
        m = pd.read_csv("results/master_regression.csv")
        piv = m.pivot_table(index="model", columns="fd", values="test_last_rmse_cap")
        fig, ax = plt.subplots(figsize=(8, 6))
        import seaborn as sns
        sns.heatmap(piv, annot=True, fmt=".1f", cmap="Blues", ax=ax)
        ax.set_title("Cross-subset comparison: test RMSE (capped RUL @125, last cycle)")
        save(fig, "figures/master_rmse_heatmap.png")
    else:
        log.warning("--only run: master heatmap NOT rebuilt (partial subset set)")
    with open("results/wallclock.json", "w") as f:
        json.dump(wall, f, indent=2)
    log.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
