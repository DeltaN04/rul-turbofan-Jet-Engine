"""Alert policy layer + simple per-engine prediction API."""
from __future__ import annotations

import numpy as np
import pandas as pd


ALERT_COLOR = {"safe": "green", "warning": "amber", "critical": "red"}
ID_TO_LABEL = {0: "safe", 1: "warning", 2: "critical"}


def rul_to_alert(rul: float) -> str:
    if rul > 50:
        return "safe"
    if rul >= 20:
        return "warning"
    return "critical"


def predict_engine(df_history: pd.DataFrame, model, feat_cols: list[str],
                   sensors: list[str]) -> dict:
    """Minimal API: pass a single engine's sensor history (must contain feat_cols).
    Returns predicted RUL + alert. Caller builds feat_cols via preprocess.feature_matrix."""
    x = df_history[feat_cols].fillna(0.0).to_numpy()[-1:]
    rul = float(model.predict(x)[0])
    alert = rul_to_alert(rul)
    return {"predicted_RUL": rul, "alert": alert, "color": ALERT_COLOR[alert]}


def apply_policy(pred_rul: np.ndarray, critical_threshold: float = 20.0,
                 warning_threshold: float = 50.0) -> np.ndarray:
    p = np.asarray(pred_rul, dtype=float)
    return np.where(p > warning_threshold, 0, np.where(p >= critical_threshold, 1, 2))
