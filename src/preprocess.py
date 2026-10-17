"""Preprocessing: sensor selection, condition normalisation, smoothing, features."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import OPS, SENSORS

log = logging.getLogger(__name__)


def variance_report(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].var(numeric_only=True).sort_values()


def select_sensors_by_variance(df: pd.DataFrame, cols: list[str], threshold: float = 1e-6) -> tuple[list[str], list[str]]:
    var = variance_report(df, cols)
    keep = var[var > threshold].index.tolist()
    drop = var[var <= threshold].index.tolist()
    return keep, drop


def smooth_per_unit(df: pd.DataFrame, cols: list[str], alpha: float = 0.3) -> pd.DataFrame:
    """Exponential smoothing applied causally within each unit."""
    out = df.copy()
    out = out.sort_values(["unit", "cycle"])
    for c in cols:
        out[c + "_sm"] = out.groupby("unit")[c].transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    return out


@dataclass
class ConditionNormalizer:
    """KMeans on op settings -> z-score sensors within cluster.

    Fitted on training data only (no leakage). For FD001/FD003 use
    global standardisation instead (see `mode`).
    """
    mode: str = "global"  # 'cluster' or 'global'
    k: int = 6
    random_state: int = 42
    kmeans: KMeans | None = None
    stats: dict = field(default_factory=dict)  # cluster -> (mean, std)
    global_mean: pd.Series | None = None
    global_std: pd.Series | None = None
    sensor_cols: list[str] = field(default_factory=list)
    op_scaler: StandardScaler | None = None  # standardises op1..3 before clustering

    def _scaled_ops(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X = df[OPS].to_numpy(dtype=float)
        if fit:
            self.op_scaler = StandardScaler().fit(X)
        assert self.op_scaler is not None
        return self.op_scaler.transform(X)

    def fit(self, df: pd.DataFrame, sensor_cols: list[str]) -> "ConditionNormalizer":
        self.sensor_cols = list(sensor_cols)
        if self.mode == "cluster":
            Z = self._scaled_ops(df, fit=True)
            self.kmeans = KMeans(n_clusters=self.k, n_init=10, random_state=self.random_state)
            labels = self.kmeans.fit_predict(Z)
            tmp = df.copy()
            tmp["_cl"] = labels
            for cl in range(self.k):
                sub = tmp.loc[tmp["_cl"] == cl, self.sensor_cols]
                mu = sub.mean()
                sd = sub.std(ddof=0)
                # Numerical-noise floor: within-condition std at float-dust level
                # (e.g. s1/s5/s19: std ~1e-13) is amplified to ±1 by z-scoring.
                # Treat as constant so these sensors normalise to ~0 and drop out.
                floor = 1e-9 * mu.abs().clip(lower=1.0)
                sd = sd.where(sd > floor, 1.0).replace(0, 1.0)
                self.stats[int(cl)] = (mu, sd)
        else:
            self.global_mean = df[self.sensor_cols].mean()
            gs = df[self.sensor_cols].std(ddof=0)
            floor = 1e-9 * self.global_mean.abs().clip(lower=1.0)
            self.global_std = gs.where(gs > floor, 1.0).replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # sensor columns can be int64 (constant integer sensors); normalisation
        # produces floats, so cast first to avoid LossySetitemError (pandas 3).
        out[self.sensor_cols] = out[self.sensor_cols].astype(float)
        if self.mode == "cluster":
            assert self.kmeans is not None
            labels = self.kmeans.predict(self._scaled_ops(out))
            out["_cl"] = labels
            for cl, (mu, sd) in self.stats.items():
                m = out["_cl"] == cl
                out.loc[m, self.sensor_cols] = (out.loc[m, self.sensor_cols] - mu.values) / sd.values
            out.drop(columns=["_cl"], inplace=True)
        else:
            assert self.global_mean is not None and self.global_std is not None
            out[self.sensor_cols] = (out[self.sensor_cols] - self.global_mean.values) / self.global_std.values
        return out


def add_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int] = [10, 30],
    include_slope: bool = True,
) -> pd.DataFrame:
    """Causal rolling mean/std (+slope) per unit. Only past data is used."""
    out = df.sort_values(["unit", "cycle"]).copy()
    g = out.groupby("unit")
    for c in cols:
        for w in windows:
            out[f"{c}_rm{w}"] = g[c].transform(lambda x: x.rolling(w, min_periods=1).mean())
            out[f"{c}_rs{w}"] = g[c].transform(lambda x: x.rolling(w, min_periods=1).std(ddof=0).fillna(0.0))
    if include_slope:
        for c in cols:
            def _slope(x: pd.Series) -> pd.Series:
                # slope of last `wmax` points via polyfit; causal
                wmax = max(windows)
                vals = x.to_numpy()
                slopes = np.zeros(len(vals))
                for i in range(len(vals)):
                    seg = vals[max(0, i - wmax + 1): i + 1]
                    if len(seg) < 2:
                        slopes[i] = 0.0
                    else:
                        xx = np.arange(len(seg))
                        slopes[i] = float(np.polyfit(xx, seg, 1)[0])
                return pd.Series(slopes, index=x.index)
            out[f"{c}_slope"] = g[c].transform(_slope)
    # NOTE: no cycle_norm feature: max cycle of a unit is its failure time,
    # unknown at prediction time (especially for truncated test engines).
    # Using it leaks the RUL label (RUL = cycle/cycle_norm - cycle).
    return out


def feature_matrix(
    df: pd.DataFrame,
    base_cols: list[str],
    windows: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    feat_df = add_rolling_features(df, base_cols, windows)
    feat_cols = [c for c in feat_df.columns if any(
        c.endswith(f"_rm{w}") or c.endswith(f"_rs{w}") or c.endswith("_slope")
        for w in windows)]
    feat_cols = base_cols + feat_cols + ["cycle"]
    # keep op settings as raw features too
    for o in OPS:
        if o in feat_df.columns and o not in feat_cols:
            feat_cols.append(o)
    X = feat_df[feat_cols].fillna(0.0)
    return X, feat_cols


def make_sequences(
    df: pd.DataFrame,
    cols: list[str],
    window: int = 30,
    target_col: str = "RUL_cap",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows per unit for deep models. Short histories are left-padded
    with the first available row (edge padding); rule stated in report."""
    Xs, ys, units = [], [], []
    for u, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        M = g[cols].to_numpy(dtype=np.float32)
        y = g[target_col].to_numpy(dtype=np.float32) if target_col in g.columns else np.zeros(len(g), dtype=np.float32)
        for i in range(len(g)):
            start = max(0, i - window + 1)
            seg = M[start: i + 1]
            if len(seg) < window:
                pad = np.repeat(seg[0:1], window - len(seg), axis=0)
                seg = np.concatenate([pad, seg], axis=0)
            Xs.append(seg)
            ys.append(y[i])
            units.append(u)
    return np.stack(Xs), np.array(ys), np.array(units)
