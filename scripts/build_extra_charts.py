"""Extra cross-cutting charts from v2 results (capped metrics, tidy classes)."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import save, setup  # noqa: E402

setup()
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; RES = ROOT / "results"


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.5); ax.axis("off")
    boxes = [
        (0.2, "Load +\nvalidate\n(26 cols)"), (1.9, "RUL labels\npiece-wise\ncap=125"),
        (3.6, "Condition norm\nFIRST\nKMeans k=6"), (5.3, "Variance drop\n(normalised)\ns1/s5…"),
        (7.0, "Smooth +\ncausal rolling\nfeat (10/30)"), (8.7, "Models: 5-fold CV\nRF/HGB/MLP\nLSTM/CNN"),
        (10.3, "Alerts\nsafe/warn/crit\nOOF-tuned t"),
    ]
    for i, (x, t) in enumerate(boxes):
        ax.text(x + 0.65, 2.4, t, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="#D6EAF8" if i < 6 else "#FADBD8", ec="#0173B2"))
        if x > 0.3:
            ax.annotate("", xy=(x + 0.05, 2.4), xytext=(x - 0.25, 2.4),
                        arrowprops=dict(arrowstyle="->", color="#0173B2", lw=1.5))
    ax.text(6, 4.1, "Turbofan RUL pipeline — normalise before smoothing; grouped 5-fold CV; test never used for selection",
            ha="center", fontsize=12, weight="bold")
    ax.text(6, 0.5, "Eval (capped primary): RMSE · MAE · R² · PHM08 · precision/recall/F1 · CM · ROC/PR · lead-time · cost · Mondrian conformal",
            ha="center", fontsize=9, style="italic")
    save(fig, str(FIG / "pipeline_block_diagram.png"))


def grouped_bars():
    reg = pd.read_csv(RES / "master_regression.csv")
    order = ["ridge", "rf", "hgb", "hgb_tuned", "mlp", "lstm", "cnn", "mean"]
    for metric, title, fname in [
        ("test_last_rmse_cap", "Test RMSE by model × subset (capped RUL @125, last cycle — lower is better)", "compare_rmse_bars.png"),
        ("test_last_phm_cap", "PHM08 score by model × subset, capped targets (late errors penalised — lower is better)", "compare_phm_bars.png"),
    ]:
        piv = reg.pivot_table(index="model", columns="fd", values=metric)
        piv = piv.reindex([m for m in order if m in piv.index])
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(piv.columns)); w = 0.8 / max(1, len(piv))
        for i, m in enumerate(piv.index):
            vals = piv.loc[m].values
            mask = ~np.isnan(vals)
            ax.bar(x[mask] + (i - len(piv) / 2) * w + w / 2, vals[mask], w, label=m)
        ax.set_xticks(x); ax.set_xticklabels(piv.columns)
        ax.set_title(title); ax.legend(ncol=4, fontsize=8)
        ax.set_ylabel(metric)
        save(fig, str(FIG / fname))


def classification_bars():
    clf = pd.read_csv(RES / "master_classification.csv")
    sub = clf[(clf.task == "3class") & (clf.approach == "direct") & (clf["class"] == "macro")]
    piv = sub.pivot_table(index="model", columns="fd", values="f1")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(piv.columns)); w = 0.8 / max(1, len(piv))
    for i, m in enumerate(piv.index):
        vals = piv.loc[m].values
        mask = ~np.isnan(vals)
        ax.bar(x[mask] + (i - len(piv) / 2) * w + w / 2, vals[mask], w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(piv.columns)
    ax.set_title("3-class macro-F1 by model × subset (direct classifiers, last cycle)")
    ax.legend(fontsize=8); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.05)
    save(fig, str(FIG / "compare_macroF1_bars.png"))
    sub2 = clf[(clf.task == "3class") & (clf["class"] == "critical")]
    piv2 = sub2.pivot_table(index=["approach", "model"], columns="fd", values="recall")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(piv2.columns)); w = 0.8 / max(1, len(piv2))
    for i, m in enumerate(piv2.index):
        vals = piv2.loc[m].values
        mask = ~np.isnan(vals)
        ax.bar(x[mask] + (i - len(piv2) / 2) * w + w / 2, vals[mask], w,
               label=f"{m[0][:7]}:{m[1]}")
    ax.set_xticks(x); ax.set_xticklabels(piv2.columns)
    ax.set_title("Critical-class recall by approach × subset (safety priority — higher is better)")
    ax.legend(fontsize=7, ncol=2); ax.set_ylabel("critical recall"); ax.set_ylim(0, 1.05)
    save(fig, str(FIG / "compare_critrecall_bars.png"))


def cost_chart():
    import json
    rows = []
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        s = json.load(open(RES / f"{fd}_summary.json"))
        rows.append({"fd": fd, "tuned_threshold": s.get("threshold", {}).get("t"),
                     "cost_default(assumed)": s.get("cost_default"),
                     "cost_tuned": s.get("cost_tuned")})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df)); w = 0.35
    ax.bar(x - w / 2, df["cost_default(assumed)"], w, label="threshold=30 (assumed 10:1 costs)", color="#0173B2")
    ax.bar(x + w / 2, df["cost_tuned"], w, label="OOF-tuned threshold", color="#DE8F05")
    ax.set_xticks(x); ax.set_xticklabels(df["fd"])
    ax.set_title("Expected maintenance cost per subset (assumed costs: missed failure=10× false alarm)")
    ax.set_ylabel("expected cost"); ax.legend()
    for i, t in enumerate(df["tuned_threshold"]):
        ax.text(i, max(df["cost_default(assumed)"].max(), df["cost_tuned"].max()) * 1.02,
                f"t={t}", ha="center", fontsize=8)
    save(fig, str(FIG / "cost_comparison.png"))
    df.to_csv(RES / "cost_comparison.csv", index=False)


def classdist_compare():
    sys.path.insert(0, str(ROOT / "src"))
    from src.data import load_split, add_rul_train, add_class_labels
    rows = {}
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        tr, _, _ = load_split(ROOT / "data" / "raw", fd)
        tr = add_class_labels(add_rul_train(tr))
        c = tr.cls3.value_counts(normalize=True)
        rows[fd] = [c.get("safe", 0), c.get("warning", 0), c.get("critical", 0)]
    df = pd.DataFrame(rows, index=["safe", "warning", "critical"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(4); w = 0.22
    for i, cls in enumerate(["safe", "warning", "critical"]):
        ax.bar(x + (i - 1) * w, df.loc[cls].values, w, label=cls,
               color=["#029E73", "#DE8F05", "#CC78BC"][i])
    ax.set_xticks(x); ax.set_xticklabels(["FD001", "FD002", "FD003", "FD004"])
    ax.set_title("Class imbalance: share of train rows per class × subset (safe dominates)")
    ax.set_ylabel("share"); ax.legend()
    save(fig, str(FIG / "classdist_compare.png"))


def ablation_charts():
    abl = pd.read_csv(RES / "ablations.csv")
    # smooth order (the A1 bug, documented as an ablation on FD002)
    sub = abl[(abl.subset == "FD002") & (abl.ablation == "smooth_order")]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sub["setting"], sub["test_value"], color=["#CC78BC", "#029E73"])
    ax.set_title("FD002 ablation: smooth-before-normalise (legacy bug) vs normalise-first (fixed)")
    ax.set_ylabel("test RMSE capped"); ax.set_xticklabels(sub["setting"], rotation=10)
    save(fig, str(FIG / "FD002_ablation_smoothorder.png"))
    # condnorm on/off FD002
    sub = abl[(abl.subset == "FD002") & (abl.ablation == "condnorm")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sub["setting"], sub["test_value"], color="#0173B2")
    ax.set_title("FD002 ablation: condition normalisation off vs on (capped RMSE)")
    ax.set_ylabel("test RMSE capped")
    save(fig, str(FIG / "FD002_ablation_condnorm.png"))
    # rolling per subset
    for fd in ["FD001", "FD002"]:
        sub = abl[(abl.subset == fd) & (abl.ablation == "rolling")]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(sub["setting"].astype(str), sub["test_value"], color="#029E73")
        ax.set_title(f"{fd} ablation: rolling-window set vs test RMSE (capped)")
        ax.set_ylabel("test RMSE capped")
        save(fig, str(FIG / f"{fd}_ablation_rolling.png"))
    # cap on common target
    for fd in ["FD001", "FD002"]:
        sub = abl[(abl.subset == fd) & (abl.ablation == "rul_cap")]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(sub["setting"].astype(str), sub["test_value"], color="#0173B2")
        ax.set_title(f"{fd} ablation: RUL cap vs test RMSE on common capped-125 target")
        ax.set_xlabel("cap"); ax.set_ylabel("RMSE (common-125 target)")
        save(fig, str(FIG / f"{fd}_ablation_cap.png"))
    # class weight (macro-F1)
    for fd in ["FD001", "FD002"]:
        sub = abl[(abl.subset == fd) & (abl.ablation == "class_weight")].copy()
        sub["setting"] = sub["setting"].fillna("None")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(sub["setting"].astype(str), sub["test_value"], color="#CC78BC")
        ax.set_title(f"{fd} ablation: class_weight vs 3-class macro-F1 (higher is better)")
        ax.set_ylabel("macro-F1")
        save(fig, str(FIG / f"{fd}_ablation_classweight.png"))
    # no_cycle / no_ops / r_vs_c
    sub = abl[abl.ablation.isin(["no_cycle", "no_ops"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    piv = sub.pivot_table(index="subset", columns="ablation", values="test_value")
    piv.plot.bar(ax=ax, color=["#DE8F05", "#0173B2"])
    ax.set_title("Ablation: dropping cycle / op-settings features (capped test RMSE)")
    ax.set_ylabel("test RMSE capped")
    save(fig, str(FIG / "ablation_cycle_ops.png"))


def conformal_chart():
    rows = []
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        p = RES / f"{fd}_conformal.txt"
        if p.exists():
            for line in open(p):
                m = re.search(r"(\d+)% Mondrian.*coverage[^=]*=([\d.]+)", line)
                if m:
                    rows.append({"fd": fd, "nominal": float(m.group(1)) / 100,
                                 "empirical": float(m.group(2))})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    for fd in df.fd.unique():
        sub = df[df.fd == fd].sort_values("nominal")
        ax.plot(sub["nominal"], sub["empirical"], marker="o", label=fd)
    ax.plot([0.85, 0.99], [0.85, 0.99], "k--", label="ideal calibration")
    ax.set_title("Mondrian conformal: nominal vs empirical coverage (capped RUL, last cycle)")
    ax.set_xlabel("nominal"); ax.set_ylabel("empirical"); ax.legend()
    save(fig, str(FIG / "conformal_calibration.png"))


def last_vs_all_cycles():
    reg = pd.read_csv(RES / "master_regression.csv")
    sub = reg[reg.model.isin(["ridge", "rf", "hgb", "mlp"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(4); fds = ["FD001", "FD002", "FD003", "FD004"]; w = 0.18
    models = [m for m in ["ridge", "rf", "hgb", "mlp"] if m in sub.model.unique()]
    for i, m in enumerate(models):
        last = [sub[(sub.fd == f) & (sub.model == m)]["test_last_rmse_cap"].values[0] for f in fds]
        allc = [sub[(sub.fd == f) & (sub.model == m)]["test_all_rmse_cap"].values[0] for f in fds]
        ax.bar(x + (i - 2) * w, last, w * 0.9, label=f"{m}-last")
    ax.set_xticks(x); ax.set_xticklabels(fds)
    ax.set_title("Last-cycle capped RMSE by model × subset (all-cycles is far harder: early-life flat region)")
    ax.set_ylabel("RMSE capped"); ax.legend(fontsize=8, ncol=2)
    save(fig, str(FIG / "last_vs_all_rmse.png"))


def deep_window_chart():
    p = RES / "deep_window_ablation.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, fd in zip(axes, ["FD001", "FD002"]):
        sub = df[df.subset == fd]
        x = np.arange(len(sub.window.unique())); w = 0.35
        for i, m in enumerate(["lstm", "cnn"]):
            s = sub[sub.model == m].sort_values("window")
            ax.bar(x + (i - 0.5) * w, s["test_rmse_cap"], w, label=m)
        ax.set_xticks(x); ax.set_xticklabels(sorted(sub.window.unique()))
        ax.set_title(f"{fd}: deep window length vs test RMSE capped")
        ax.set_xlabel("window"); ax.set_ylabel("test RMSE capped"); ax.legend()
    save(fig, str(FIG / "deep_window_ablation.png"))


if __name__ == "__main__":
    pipeline_diagram(); grouped_bars(); classification_bars(); cost_chart()
    classdist_compare(); ablation_charts(); conformal_chart(); last_vs_all_cycles()
    deep_window_chart()
    print("extra charts done:", len(list(FIG.glob('*.png'))), "figures total")
