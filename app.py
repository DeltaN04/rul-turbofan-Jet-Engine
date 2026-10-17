"""Streamlit demo: pick a test engine -> predicted RUL + alert + trajectory."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.preprocess import feature_matrix, smooth_per_unit  # noqa: E402

st.set_page_config(page_title="Turbofan RUL & Maintenance Alert Demo", layout="wide")
st.title("Turbofan Engine RUL & Maintenance Alert Demo")
st.caption("C-MAPSS case study — TE7483. Causal rolling features; no future leakage.")

fd = st.sidebar.selectbox("Subset", ["FD001", "FD002", "FD003", "FD004"])
models = sorted((ROOT / "models").glob(f"{fd}_*.pkl"))
if not models:
    st.warning("No trained models found yet — run `python run_pipeline.py` first.")
    st.stop()
mpath = st.sidebar.selectbox("Model file", [m.name for m in models])
with open(ROOT / "models" / mpath, "rb") as f:
    bundle = pickle.load(f)
model, feat_cols = bundle["model"], bundle["feat_cols"]
sensors, norm, windows = bundle["sensors"], bundle["norm"], bundle["windows"]
scaler, alpha = bundle["scaler"], bundle.get("smooth_alpha", 0.3)

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
test = pd.read_csv(ROOT / "data" / "raw" / f"test_{fd}.txt", sep=r"\s+", header=None)
if test.shape[1] == 27:
    test = test.iloc[:, :26]
test.columns = COLS
units = sorted(test["unit"].unique())
u = st.sidebar.selectbox("Engine", units)
n_hist = st.sidebar.slider("Recent history shown (cycles)", 10, 300, 80)

hist = test[test.unit == u].sort_values("cycle")
# fixed pipeline order (A1): condition-normalise FIRST, then smooth normalised
# sensors (smoothing raw multi-condition data blends operating modes).
sm_n = norm.transform(hist.copy())
sm = smooth_per_unit(sm_n, sensors, alpha)
for c in sensors:
    sm_n[c] = sm[c + "_sm"]
Xf, _ = feature_matrix(sm_n, sensors, windows)
# align columns, scale
Xf = Xf[feat_cols].fillna(0.0)
Xs = scaler.transform(Xf.values)
pred_all = model.predict(Xs)
rul = float(pred_all[-1])
alert = "safe" if rul > 50 else ("warning" if rul >= 20 else "critical")
color = {"safe": "green", "warning": "orange", "critical": "red"}[alert]

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL (cycles)", f"{rul:.1f}")
c2.markdown(f"### Alert: :{color}[{alert.upper()}]")
c3.metric("Cycles observed", f"{len(hist)}")
st.markdown("**Policy:** green = safe (RUL>50, routine monitoring) · amber = warning "
            "(20–50, schedule inspection) · red = critical (<20, ground & inspect now). "
            "Costs assumed: missed failure = 10× false alarm. Human-in-the-loop required.")
chart = pd.DataFrame({"cycle": hist["cycle"].values, "pred_RUL": pred_all})
st.line_chart(chart.set_index("cycle"))
st.dataframe(hist.tail(10))
