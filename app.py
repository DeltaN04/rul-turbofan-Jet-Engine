"""Streamlit demo: Turbofan RUL & Maintenance Alert Dashboard.

Pages:
1. Overview - project summary, headline metrics, subset selector
2. Engine Monitor - per-engine prediction with replay slider, conformal intervals, alerts
3. Fleet View - all test engines table with filters, alert counts
4. Model Performance - regression/classification tables, confusion matrices, PR curves, lead-time
5. Cost and Alert Policy - interactive cost ratio slider, threshold recommendations
6. About / Limitations - caveats, validation protocol, AI disclosure
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.alerts import rul_to_alert, ALERT_COLOR  # noqa: E402
from src.data import add_rul_test_all_cycles, load_split, add_class_labels  # noqa: E402
from src.preprocess import ConditionNormalizer, feature_matrix, smooth_per_unit  # noqa: E402

st.set_page_config(page_title="Turbofan RUL & Maintenance Alert Dashboard", layout="wide")

# ============================================================
# Colour-blind-safe palette
# ============================================================
PALETTE = {
    "safe": "#0072B2",       # blue
    "warning": "#E69F00",    # orange
    "critical": "#D55E00",   # vermillion
    "true": "#009E73",       # green
    "pred": "#CC79A7",       # reddish purple
    "bg": "#F5F5F5",
    "text": "#333333",
}

# Streamlit's :color[text] markdown only accepts named colours.
BADGE_COLOR = {"safe": "green", "warning": "orange", "critical": "red"}

# ============================================================
# Cached loaders
# ============================================================
@st.cache_resource(show_spinner="Loading model bundle…")
def load_model_bundle(fd: str):
    """Load the deployed model pickle for a subset."""
    model_path = ROOT / "models" / f"{fd}_hgb.pkl"
    if not model_path.exists():
        st.error(f"Model file not found: {model_path}. Run `python run_pipeline.py` first.")
        st.stop()
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


@st.cache_data(show_spinner="Loading test data…")
def load_test_data(fd: str):
    """Load raw test data and RUL labels, compute true RUL for all cycles."""
    raw_dir = ROOT / "data" / "raw"
    train_raw, test_raw, rul_last = load_split(str(raw_dir), fd)
    test_all = add_rul_test_all_cycles(test_raw, rul_last)
    test_all = add_class_labels(test_all, col="RUL")
    return test_all


@st.cache_data(show_spinner="Loading summary metrics…")
def load_summary(fd: str):
    """Load per-subset summary JSON."""
    p = ROOT / "results" / f"{fd}_summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading master tables…")
def load_master_tables():
    """Load master regression and classification CSVs."""
    reg = pd.read_csv(ROOT / "results" / "master_regression.csv")
    clf = pd.read_csv(ROOT / "results" / "master_classification.csv")
    return reg, clf


@st.cache_data(show_spinner="Loading conformal intervals…")
def load_conformal(fd: str):
    """Load Mondrian conformal intervals (bin half-widths)."""
    p = ROOT / "results" / f"{fd}_conformal_bins.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner="Loading cost sensitivity…")
def load_cost_sensitivity(fd: str):
    """Load cost sensitivity table for threshold tuning."""
    p = ROOT / "results" / f"{fd}_cost_sensitivity.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner="Loading lead time data…")
def load_leadtime(fd: str, model_name: str):
    """Load lead time analysis CSV."""
    p = ROOT / "results" / f"{fd}_leadtime_{model_name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner="Loading classification CIs…")
def load_clf_cis(fd: str):
    """Load per-class bootstrap CIs."""
    p = ROOT / "results" / f"{fd}_classification_cis.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def get_conformal_half_width(fd: str, pred_rul: float, level: str = "90") -> float:
    """Get conformal half-width for a predicted RUL value."""
    df = load_conformal(fd)
    if df.empty:
        return float("nan")
    level = str(level)
    row = df[(df.level == level) & (df.true_rul_bin.apply(lambda b: _in_bin(pred_rul, b)))]
    if row.empty:
        # fallback to global
        row = df[df.level == level]
    if row.empty:
        return float("nan")
    return float(row.half_width.iloc[0])


def _in_bin(val: float, bin_str: str) -> bool:
    """Check if value falls in bin string like '0-20' or '50-125'."""
    try:
        lo, hi = map(float, bin_str.split("-"))
        return lo <= val <= hi
    except Exception:
        return False


def preprocess_engine_history(hist: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Apply the exact preprocessing pipeline to a single engine's history."""
    sensors = bundle["sensors"]
    norm = bundle["norm"]
    windows = bundle["windows"]
    feat_cols = bundle["feat_cols"]
    scaler = bundle["scaler"]
    alpha = bundle.get("smooth_alpha", 0.3)

    # Ensure correct column order and types
    hist = hist.sort_values("cycle").copy()

    # 1. Condition normalise (fit on train, transform test)
    sm_n = norm.transform(hist.copy())

    # 2. Causal exponential smoothing on normalised sensors
    sm = smooth_per_unit(sm_n, sensors, alpha)
    for c in sensors:
        sm_n[c] = sm[c + "_sm"]

    # 3. Causal rolling features
    Xf, _ = feature_matrix(sm_n, sensors, windows)

    # 4. Align columns and scale
    Xf = Xf[feat_cols].fillna(0.0)
    Xs = scaler.transform(Xf.values)
    return Xs


# ============================================================
# Sidebar navigation
# ============================================================
PAGES = [
    "1. Overview",
    "2. Engine Monitor",
    "3. Fleet View",
    "4. Model Performance",
    "5. Cost & Alert Policy",
    "6. About / Limitations",
]
page = st.sidebar.radio("Navigation", PAGES, index=0, key="navigation")

# Subset selector (shared across pages)
fd = st.sidebar.selectbox("Subset", ["FD001", "FD002", "FD003", "FD004"], index=0)

# Load shared resources
bundle = load_model_bundle(fd)
test_all = load_test_data(fd)
summary = load_summary(fd)
model = bundle["model"]
feat_cols = bundle["feat_cols"]
sensors = bundle["sensors"]
norm = bundle["norm"]
windows = bundle["windows"]
scaler = bundle["scaler"]

units = sorted(test_all["unit"].unique())
n_test_engines = len(units)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "1. Overview":
    st.title("Turbofan Engine RUL Estimation & Maintenance Alerting")
    st.caption("NASA C-MAPSS (PHM08) · TE7483 Applications and Use Cases of Machine Learning")

    th = summary.get("threshold", {})
    lt = summary.get("lead_time", {})
    reg_all, _ = load_master_tables()
    hgb_row = reg_all[(reg_all.fd == fd) & (reg_all.model == "hgb")]
    test_rmse = float(hgb_row["test_last_rmse_cap"].iloc[0]) if not hgb_row.empty else float("nan")
    test_phm = float(hgb_row["test_last_phm_cap"].iloc[0]) if not hgb_row.empty else float("nan")
    recall = float(th.get("test_recall", float("nan")))
    precision = float(th.get("test_precision", float("nan")))
    warn_lead = float(lt.get("warn_median", float("nan")))

    st.markdown(f"""
    **Problem.** Unscheduled engine removals cost airlines millions per year. This project predicts
    **Remaining Useful Life (RUL)** in cycles from multivariate sensor histories and converts RUL
    into an actionable **green/amber/red maintenance alert** policy.

    **Method (one paragraph).** We engineer causal rolling-window features (mean, std, slope) on
    operating-condition-normalised sensors (KMeans z-score for multi-condition subsets FD002/FD004),
    train HistGradientBoostingRegressor (CV-selected via grouped 5-fold by engine) with RUL capped at
    125 cycles, and tune alert thresholds on out-of-fold predictions at realistic truncation points
    (frozen before test). Deep baselines (1D-CNN, LSTM) use length-30 causal windows with a by-engine
    holdout. Conformal prediction (Mondrian, split) provides prediction intervals. All fits are
    leakage-free: scalers, KMeans, selectors fit on training folds only; test features use past data
    of that engine only.

    **Key results ({fd}, deployed HistGradientBoosting):** test RMSE **{test_rmse:.2f} cycles**
    (capped @125), PHM08 **{test_phm:,.0f}**, fails-within-30 recall **{recall:.2f}** at precision
    **{precision:.2f}**, median warning lead **{warn_lead:.0f} cycles**.
    """)

    st.subheader("Headline Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test RMSE (capped)", f"{test_rmse:.2f}")
    c2.metric("Fails≤30 Recall (tuned)", f"{recall:.2f}")
    c3.metric("Median Warning Lead", f"{warn_lead:.0f} cycles")
    c4.metric("Test Engines", n_test_engines)

    st.subheader("Subset Sizes (last cycle per engine)")
    sizes_df = pd.DataFrame([{
        "FD": f,
        "Train Engines": json.load(open(ROOT / "results" / f"{f}_summary.json")).get("summary", {}).get("n_train_units"),
        "Test Engines": json.load(open(ROOT / "results" / f"{f}_summary.json")).get("summary", {}).get("n_test_units"),
    } for f in ["FD001", "FD002", "FD003", "FD004"]])
    st.dataframe(sizes_df, use_container_width=True, hide_index=True)

    st.info("Use the sidebar to navigate to **Engine Monitor** for per-engine predictions, "
            "**Fleet View** for a sortable table of all test engines, "
            "**Model Performance** for detailed metrics and figures, "
            "**Cost & Alert Policy** for interactive threshold tuning, and "
            "**About / Limitations** for caveats.")

# ============================================================
# PAGE 2: ENGINE MONITOR
# ============================================================
elif page == "2. Engine Monitor":
    st.title("Engine Monitor — Real-time RUL & Alert Replay")
    st.caption(f"Subset: {fd} | Model: HGB (deployed) | {n_test_engines} test engines")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("Engine Selection")
        source = st.radio("Data source", ["Test set engine", "Upload CSV"], horizontal=True)

        if source == "Test set engine":
            u = st.selectbox("Engine ID", units, index=0)
            hist = test_all[test_all.unit == u].sort_values("cycle").copy()
            true_rul_last = float(hist["RUL"].iloc[-1])
            true_alert = rul_to_alert(true_rul_last)
            st.caption(f"True RUL (last cycle): **{true_rul_last:.1f}** → **{true_alert.upper()}**")
        else:
            uploaded = st.file_uploader("Upload engine history (CSV, 26 cols: unit,cycle,op1,op2,op3,s1..s21)", type="csv")
            if uploaded is None:
                st.info("Upload a CSV to proceed.")
                st.stop()
            try:
                hist = pd.read_csv(uploaded)
                expected_cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
                if list(hist.columns) != expected_cols:
                    st.error(f"CSV must have exactly 26 columns: {expected_cols}. Got: {list(hist.columns)}")
                    st.stop()
                hist = hist.sort_values("cycle")
                true_rul_last = None
                true_alert = None
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.stop()

        # Current cycle slider
        max_cycle = int(hist["cycle"].max())
        current_cycle = st.slider("Current cycle (replay)", min_value=1, max_value=max_cycle, value=max_cycle)
        hist_trim = hist[hist["cycle"] <= current_cycle].copy()

        if len(hist_trim) == 0:
            st.warning("No data up to this cycle.")
            st.stop()

        # Preprocess and predict
        Xs = preprocess_engine_history(hist_trim, bundle)
        pred_full = model.predict(Xs)
        pred_rul = float(pred_full[-1])
        alert = rul_to_alert(pred_rul)

        # Conformal interval
        conf_90 = get_conformal_half_width(fd, pred_rul, "90")
        conf_95 = get_conformal_half_width(fd, pred_rul, "95")

        st.markdown("---")
        st.subheader("Prediction")
        st.metric("Predicted RUL", f"{pred_rul:.1f} cycles")
        st.markdown(f"### Alert: :{BADGE_COLOR[alert]}[{alert.upper()}]")
        st.metric("Cycles Observed", len(hist_trim))

        if conf_90 == conf_90:  # not NaN
            st.caption(f"90% Conformal interval: [{pred_rul - conf_90:.1f}, {pred_rul + conf_90:.1f}]")
        if conf_95 == conf_95:
            st.caption(f"95% Conformal interval: [{pred_rul - conf_95:.1f}, {pred_rul + conf_95:.1f}]")

    with col_right:
        st.subheader("RUL Trajectory")
        chart_df = pd.DataFrame({
            "cycle": hist_trim["cycle"].values,
            "Predicted RUL": pred_full,
            "True RUL": hist_trim["RUL"].values if "RUL" in hist_trim.columns else np.nan,
        })

        import altair as alt
        alt.themes.enable("none")

        base = alt.Chart(chart_df).encode(
            alt.X("cycle:Q", title="Cycle", axis=alt.Axis(grid=False)),
        )

        pred_line = base.mark_line(color=PALETTE["pred"], strokeWidth=2).encode(
            alt.Y("Predicted RUL:Q", title="RUL (cycles)", scale=alt.Scale(zero=False))
        )

        true_line = base.mark_line(color=PALETTE["true"], strokeWidth=2, strokeDash=[4, 4]).encode(
            alt.Y("True RUL:Q")
        )

        # Threshold lines
        threshold_df = pd.DataFrame({
            "y": [50, 20],
            "label": ["Warning (50)", "Critical (20)"],
            "color": [PALETTE["warning"], PALETTE["critical"]],
        })
        thresholds = alt.Chart(threshold_df).mark_rule(strokeWidth=1.5).encode(
            y="y:Q",
            color=alt.Color("color:N", scale=None),
            tooltip=["label", "y"]
        )

        chart = (pred_line + true_line + thresholds).properties(
            width=700, height=350
        ).configure_axis(
            labelFontSize=12, titleFontSize=13
        ).configure_legend(
            labelFontSize=11, titleFontSize=12
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption("Solid line = predicted RUL, dashed = true RUL (if available). "
                   "Horizontal rules = alert thresholds (warning=50, critical=20). "
                   "Alert status updates with the slider.")

        # Top sensor trends (same normalisation + smoothing as the model input)
        st.subheader("Top Sensor Trends (last 50 cycles)")
        top_sensors = sensors[:6]  # first 6 kept sensors
        trend_src = norm.transform(hist_trim.copy())
        trend_sm = smooth_per_unit(trend_src, top_sensors, bundle.get("smooth_alpha", 0.3))
        for c in top_sensors:
            trend_src[c] = trend_sm[c + "_sm"]
        trend_df = trend_src.tail(50)[["cycle"] + top_sensors].copy()
        trend_long = trend_df.melt(id_vars="cycle", var_name="sensor", value_name="value")
        trend_chart = alt.Chart(trend_long).mark_line(strokeWidth=1.5).encode(
            x="cycle:Q",
            y=alt.Y("value:Q", title="Normalised z-score"),
            color="sensor:N",
            tooltip=["sensor", "cycle", "value"]
        ).properties(width=700, height=250)
        st.altair_chart(trend_chart, use_container_width=True)
        st.caption(f"First {len(top_sensors)} kept sensors — condition-normalised, causal-smoothed (last 50 cycles).")

# ============================================================
# PAGE 3: FLEET VIEW
# ============================================================
elif page == "3. Fleet View":
    st.title("Fleet View — All Test Engines")
    st.caption(f"Subset: {fd} | {n_test_engines} engines | Last-cycle predictions")

    # Compute predictions for all engines (cached)
    @st.cache_data(show_spinner="Predicting all test engines…")
    def predict_all_engines(fd: str, _bundle: dict, test_all: pd.DataFrame):
        rows = []
        for u in sorted(test_all["unit"].unique()):
            hist = test_all[test_all.unit == u].sort_values("cycle")
            Xs = preprocess_engine_history(hist, _bundle)
            pred = float(model.predict(Xs)[-1])
            alert = rul_to_alert(pred)
            true_rul = float(hist["RUL"].iloc[-1]) if "RUL" in hist.columns else None
            true_alert = rul_to_alert(true_rul) if true_rul is not None else None
            rows.append({
                "Engine": u,
                "Pred RUL": pred,
                "Pred Alert": alert,
                "True RUL": true_rul,
                "True Alert": true_alert,
                "Cycles": len(hist),
            })
        return pd.DataFrame(rows)

    fleet_df = predict_all_engines(fd, bundle, test_all)

    # Filters
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        alert_filter = st.multiselect(
            "Filter by predicted alert",
            options=["safe", "warning", "critical"],
            default=["safe", "warning", "critical"],
        )
    with col_f2:
        sort_by = st.selectbox("Sort by", ["Engine", "Pred RUL", "Pred Alert", "True RUL"])

    filtered = fleet_df[fleet_df["Pred Alert"].isin(alert_filter)].copy()
    filtered = filtered.sort_values(sort_by)

    # Style alerts with colors
    def style_alert(val):
        color = PALETTE.get(val, "#333333")
        return f"background-color: {color}; color: white; font-weight: bold; border-radius: 4px; padding: 2px 6px;"

    styled = filtered.style.format({"Pred RUL": "{:.1f}", "True RUL": "{:.1f}"}).map(
        style_alert, subset=["Pred Alert"]
    )
    if "True Alert" in filtered.columns:
        styled = styled.map(style_alert, subset=["True Alert"])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    # Alert counts bar
    st.subheader("Alert Counts")
    counts = filtered["Pred Alert"].value_counts().reindex(["safe", "warning", "critical"], fill_value=0)
    count_df = pd.DataFrame({"Alert": counts.index, "Count": counts.values})
    count_df["Color"] = count_df["Alert"].map(PALETTE)

    import altair as alt
    bar = alt.Chart(count_df).mark_bar().encode(
        x=alt.X("Alert:N", sort=["safe", "warning", "critical"]),
        y="Count:Q",
        color=alt.Color("Color:N", scale=None),
        tooltip=["Alert", "Count"]
    ).properties(width=400, height=250)
    st.altair_chart(bar, use_container_width=True)
    st.caption("Distribution of predicted alerts across the filtered fleet.")

# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================
elif page == "4. Model Performance":
    st.title("Model Performance")
    st.caption(f"Subset: {fd} | All metrics from results/ (no recomputation)")

    reg_master, clf_master = load_master_tables()
    fd_reg = reg_master[reg_master.fd == fd].copy()
    fd_clf = clf_master[clf_master.fd == fd].copy()

    # Model selector
    models_available = sorted(fd_reg["model"].unique())
    selected_model = st.selectbox("Model", models_available, index=models_available.index("hgb") if "hgb" in models_available else 0)

    # --- Regression table ---
    st.subheader("Regression Metrics (Capped RUL @125, Last Cycle)")
    reg_sel = fd_reg[fd_reg.model == selected_model]
    if not reg_sel.empty:
        cols_show = ["model", "cv_rmse_mean", "cv_rmse_std", "cv_phm_mean",
                     "test_last_rmse_cap", "test_last_mae_cap", "test_last_phm_cap",
                     "test_last_rmse_true", "test_last_mae_true", "test_last_phm_true"]
        cols_show = [c for c in cols_show if c in reg_sel.columns]
        # Format only numeric columns
        numeric_cols = reg_sel[cols_show].select_dtypes(include=["number"]).columns
        format_dict = {c: "{:.2f}" for c in numeric_cols}
        st.dataframe(reg_sel[cols_show].style.format(format_dict), use_container_width=True, hide_index=True)

    # --- Classification table ---
    st.subheader("Classification Metrics (Last Cycle)")
    clf_sel = fd_clf[(fd_clf.model == selected_model) & (fd_clf["class"].isin(["safe", "warning", "critical", "positive", "macro"]))]
    if not clf_sel.empty:
        cols_show = ["task", "approach", "class", "precision", "recall", "f1", "support"]
        cols_show = [c for c in cols_show if c in clf_sel.columns]
        st.dataframe(clf_sel[cols_show].style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}), use_container_width=True, hide_index=True)

    # --- Confusion Matrices ---
    st.subheader("Confusion Matrices")
    import altair as alt
    for task, task_name in [("3class", "3-Class"), ("binary30", "Binary ≤30")]:
        clf_cv = json.load(open(ROOT / "results" / f"{fd}_clf_cv.json")) if (ROOT / "results" / f"{fd}_clf_cv.json").exists() else {}
        head = clf_cv.get("head3" if task == "3class" else "headb", "hgb")
        cm_path = ROOT / "figures" / f"{fd}_cm_{'3class' if task == '3class' else 'binary'}_{head}.png"
        cm_norm_path = ROOT / "figures" / f"{fd}_cm_{'3class' if task == '3class' else 'binary'}_{head}_norm.png"
        if cm_path.exists():
            st.image(str(cm_path), caption=f"{task_name} Confusion Matrix ({head} direct)", width=400)
        if cm_norm_path.exists():
            st.image(str(cm_norm_path), caption=f"{task_name} Row-Normalised ({head} direct)", width=400)

    # --- PR / ROC Curves ---
    st.subheader("Precision–Recall & ROC Curves (Binary ≤30)")
    clf_cv = json.load(open(ROOT / "results" / f"{fd}_clf_cv.json")) if (ROOT / "results" / f"{fd}_clf_cv.json").exists() else {}
    headb = clf_cv.get("headb", "hgb")
    pr_path = ROOT / "figures" / f"{fd}_pr_{headb}.png"
    roc_path = ROOT / "figures" / f"{fd}_roc_{headb}.png"
    c1, c2 = st.columns(2)
    if pr_path.exists():
        c1.image(str(pr_path), caption=f"PR Curve ({headb})", width=400)
    if roc_path.exists():
        c2.image(str(roc_path), caption=f"ROC Curve ({headb})", width=400)

    # --- Lead Time Distribution ---
    st.subheader("Lead Time Distribution")
    lt_df = load_leadtime(fd, selected_model)
    if not lt_df.empty:
        import altair as alt
        warn_lead = lt_df["warn_lead"].dropna()
        crit_lead = lt_df["crit_lead"].dropna()
        lead_df = pd.DataFrame({
            "lead": list(warn_lead) + list(crit_lead),
            "type": ["warning"] * len(warn_lead) + ["critical"] * len(crit_lead)
        })
        hist_chart = alt.Chart(lead_df).mark_bar(opacity=0.7).encode(
            x=alt.X("lead:Q", bin=alt.Bin(maxbins=20), title="Lead Time (cycles)"),
            y="count()",
            color=alt.Color("type:N", scale=alt.Scale(domain=["warning", "critical"], range=[PALETTE["warning"], PALETTE["critical"]])),
            tooltip=["type", "count()"]
        ).properties(width=700, height=300)
        st.altair_chart(hist_chart, use_container_width=True)
        st.caption("Cycles of advance warning before true failure when alert first raised.")
    else:
        st.info("Lead time data not available for this model/subset.")

    # --- Classification CIs ---
    st.subheader("Bootstrap 95% CIs (Per-Class Recall/Precision)")
    cis_df = load_clf_cis(fd)
    if not cis_df.empty:
        cis_fd = cis_df[cis_df.model == selected_model]
        if not cis_fd.empty:
            # The CIs file has columns: fd, task, approach, model, metric, mean, lo, hi
            # No 'class' column - metric encodes class info (e.g., critical_recall, positive_f1)
            display_cols = [c for c in ["task", "approach", "metric", "mean", "lo", "hi"] if c in cis_fd.columns]
            format_dict = {c: "{:.3f}" for c in ["mean", "lo", "hi"] if c in cis_fd.columns}
            st.dataframe(cis_fd[display_cols].style.format(format_dict), use_container_width=True, hide_index=True)
        else:
            st.info("No CIs for selected model.")
    else:
        st.info("Classification CIs not found.")

# ============================================================
# PAGE 5: COST & ALERT POLICY
# ============================================================
elif page == "5. Cost & Alert Policy":
    st.title("Cost & Alert Policy")
    st.caption(f"Subset: {fd} | Costs are assumed (not airline data) | Thresholds frozen from OOF")

    # Load cost sensitivity data
    cost_df = load_cost_sensitivity(fd)
    summary_fd = load_summary(fd)
    th = summary_fd.get("threshold", {})

    st.subheader("Current Frozen Threshold")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frozen Threshold (t)", f"{th.get('t', 'N/A'):.1f}")
    c2.metric("Test Recall", f"{th.get('test_recall', 'N/A'):.3f}")
    c3.metric("Test Precision", f"{th.get('test_precision', 'N/A'):.3f}")
    c4.metric("Rule", th.get('rule_used', 'N/A')[:30])

    st.markdown("---")
    st.subheader("Interactive Cost Ratio")
    st.markdown("Adjust the **missed-failure : false-alarm** cost ratio. The table below shows "
                "the OOF-optimal threshold and the resulting test-set cost/recall/precision "
                "for the deployed model.")

    ratio = st.slider("Cost ratio (missed failure : false alarm)", 1, 100, 10, step=1)

    if not cost_df.empty:
        # Find closest ratio row
        cost_df["ratio_diff"] = (cost_df["fn_fp_ratio"] - ratio).abs()
        best_row = cost_df.loc[cost_df["ratio_diff"].idxmin()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OOF Optimal t", f"{best_row['t_opt_oof']:.1f}")
        c2.metric("Test Cost", f"{best_row['test_cost']:.0f}")
        c3.metric("Test Recall", f"{best_row['test_recall']:.3f}")
        c4.metric("Test Precision", f"{best_row['test_precision']:.3f}")

        st.dataframe(cost_df[["fn_fp_ratio", "t_opt_oof", "test_cost", "test_recall", "test_precision"]].style.format({
            "t_opt_oof": "{:.1f}", "test_cost": "{:.0f}", "test_recall": "{:.3f}", "test_precision": "{:.3f}"
        }), use_container_width=True, hide_index=True)

        # Plot cost vs threshold
        import altair as alt
        cost_plot = alt.Chart(cost_df).mark_line(point=True).encode(
            x="fn_fp_ratio:Q",
            y="test_cost:Q",
            tooltip=["fn_fp_ratio", "t_opt_oof", "test_cost", "test_recall", "test_precision"]
        ).properties(width=700, height=300)
        st.altair_chart(cost_plot, use_container_width=True)
        st.caption("Test expected cost vs cost ratio. The threshold is re-optimised on OOF for each ratio.")
    else:
        st.warning("Cost sensitivity table not found. Run pipeline with ablations.")

    st.markdown("---")
    st.subheader("Threshold Tuning Details")
    st.markdown("""
    - **How it works:** Out-of-fold RUL predictions from the deployed model at random truncation
      points per training engine → compute precision/recall/cost for candidate thresholds t ∈ [10, 60]
      → pick threshold minimising cost subject to OOF recall ≥ 0.95 (or min cost if infeasible).
    - **Frozen:** The chosen threshold is fixed *before* evaluating on the test set.
    - **Costs assumed:** missed failure = 10× false alarm (sensitivity shown above).
    - **Target met:** {} on OOF.
    """.format("✅ Yes" if th.get("target_met") else "❌ No"))

    # Show threshold tuning figure
    thresh_fig = ROOT / "figures" / f"{fd}_thresh_tuning.png"
    if thresh_fig.exists():
        st.image(str(thresh_fig), caption="OOF Precision–Recall Trade-off (Validation Only)", width=700)

# ============================================================
# PAGE 6: ABOUT / LIMITATIONS
# ============================================================
elif page == "6. About / Limitations":
    st.title("About / Limitations")
    st.caption("Important caveats and methodology notes")

    st.markdown("""
    ### 🔬 Simulated Data Caveat
    The NASA C-MAPSS dataset is a **physics-based simulation**, not real engine data.
    - No sensor faults, drift, or dropouts
    - No maintenance censoring (engines run to failure in train)
    - Operating conditions are discrete and known (6 modes in FD002/FD004)
    - Test sets are small: 100–259 engines per subset; only 13 critical engines on FD001/FD003
    - All confidence intervals are wide; claims carry uncertainty

    ### ✅ Validation Protocol (Leakage-Free)
    - **Grouped 5-fold CV** by engine for all sklearn models (Ridge, RF, HGB, MLP)
    - **By-engine 20% holdout** for deep models (LSTM, CNN)
    - **All preprocessors fit on training folds only:** ConditionNormalizer (KMeans), sensor variance
      selector, exponential smoother, StandardScaler
    - **Test features use only past data** of that engine (causal rolling windows; short histories
      edge-padded with first row)
    - **Model selection uses CV scores only** — test set never touches model choice
    - **Thresholds tuned on OOF predictions** at truncation points, frozen before test
    - **Ablations use same protocol** (CV + frozen test value)
    - **Conformal calibration** uses OOF residuals at truncation points

    ### 👷 Human-in-the-Loop (Safety)
    - **Alerts advise; humans decide** — every red alert requires engineer sign-off
    - Grounding is never autonomous
    - False-alarm budgets and subgroup checks per condition/fault are part of deployment policy
    - All costs in this report are **assumed** (10:1), not from airline data

    ### 🤖 AI Assistance Disclosure
    This project was developed with assistance from AI coding assistants (OpenCode and other tools).
    All code, experiments, and text were reviewed and approved by the human authors.
    The AI was used for: code scaffolding, debugging, report/slide generation, and test writing.
    Final scientific judgments, method choices, and interpretations are the authors' responsibility.

    ### 📦 Reproducibility
    ```bash
    pip install -r requirements.txt
    python run_pipeline.py --fresh
    streamlit run app.py
    ```
    All artefacts (results/, figures/, models/) are regenerated from scratch.
    """)

    st.subheader("Key Files")
    st.markdown("""
    | File | Purpose |
    |------|---------|
    | `run_pipeline.py` | End-to-end pipeline (EDA → features → models → ablations) |
    | `src/data.py` | Loading, RUL computation, class labels |
    | `src/preprocess.py` | Condition normalisation, smoothing, rolling features |
    | `src/train.py` | Grouped CV, tuning, OOF predictions, threshold tuning |
    | `src/evaluate.py` | Metrics (RMSE, PHM, classification, lead-time, bootstrap CIs) |
    | `src/alerts.py` | Alert policy (RUL → green/amber/red) |
    | `scripts/build_report.py` | Generates report.docx from results/ |
    | `scripts/build_slides.py` | Generates presentation.pptx from results/ |
    | `app.py` | This Streamlit dashboard |
    """)

# ============================================================
# Footer
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("Turbofan RUL Dashboard · TE7483 · C-MAPSS PHM08")
st.sidebar.caption("`streamlit run app.py`")