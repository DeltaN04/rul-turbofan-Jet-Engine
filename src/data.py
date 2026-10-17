"""Loading, validation and labelling for NASA C-MAPSS."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]
OPS = ["op1", "op2", "op3"]


def load_split(raw_dir: str | Path, fd: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train/test/RUL files for one subset (e.g. fd='FD001').

    Parses whitespace-separated files with no header and validates 26 columns.
    """
    raw = Path(raw_dir)
    train = pd.read_csv(raw / f"train_{fd}.txt", sep=r"\s+", header=None)
    test = pd.read_csv(raw / f"test_{fd}.txt", sep=r"\s+", header=None)
    # trailing-space files can produce a 27th all-NaN column -> drop it
    for name, df in (("train", train), ("test", test)):
        if df.shape[1] == 27 and df.iloc[:, 26].isna().all():
            df.drop(columns=df.columns[26], inplace=True)
        if df.shape[1] != 26:
            raise ValueError(f"{name}_{fd}: expected 26 cols, got {df.shape[1]}")
        # drop any all-NaN column defensively
        all_nan = [c for c in df.columns if df[c].isna().all()]
        if all_nan:
            raise ValueError(f"{name}_{fd}: all-NaN columns {all_nan}")
    train.columns = COLS
    test.columns = COLS
    for df in (train, test):
        df["unit"] = df["unit"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
    rul = np.loadtxt(raw / f"RUL_{fd}.txt", dtype=float)
    n_units = int(test["unit"].nunique())
    if len(rul) != n_units:
        raise ValueError(f"RUL_{fd}: len={len(rul)} != test units={n_units}")
    # RUL file is in unit order (unit ids are 1..N contiguous in test files)
    log.info("%s: train=%s test=%s units(train/test)=%d/%d", fd, train.shape, test.shape,
             train["unit"].nunique(), n_units)
    return train, test, rul


def add_rul_train(train: pd.DataFrame) -> pd.DataFrame:
    """Run-to-failure RUL: max_cycle_of_unit - cycle."""
    out = train.copy()
    out["RUL"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    return out


def add_rul_test_all_cycles(test: pd.DataFrame, rul_last: np.ndarray) -> pd.DataFrame:
    """True RUL for EVERY test row: RUL_last(unit) + (last_cycle - cycle)."""
    out = test.copy()
    units = sorted(out["unit"].unique())
    assert len(units) == len(rul_last), "RUL length mismatch"
    unit_to_rul = {u: float(r) for u, r in zip(units, rul_last)}
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL_last"] = out["unit"].map(unit_to_rul)
    out["RUL"] = out["RUL_last"] + (last_cycle - out["cycle"])
    return out


def add_piecewise_rul(df: pd.DataFrame, cap: float = 125.0, col: str = "RUL") -> pd.DataFrame:
    out = df.copy()
    out["RUL_cap"] = np.minimum(out[col].to_numpy(dtype=float), cap)
    return out


def add_class_labels(df: pd.DataFrame, col: str = "RUL") -> pd.DataFrame:
    """3-class + binary labels from TRUE uncapped RUL.

    Convention (documented, tested): safe RUL>50, warning 20<=RUL<=50,
    critical RUL<20; binary fails-within-30 = (RUL<=30).
    So RUL=50->warning, 20->warning, 19->critical, 51->safe, 30->positive.
    """
    out = df.copy()
    r = out[col].to_numpy(dtype=float)
    cls = np.where(r > 50, "safe", np.where(r >= 20, "warning", "critical"))
    out["cls3"] = cls
    out["cls3_id"] = np.where(r > 50, 0, np.where(r >= 20, 1, 2))
    out["bin30"] = (r <= 30).astype(int)
    return out


def dataset_summary(train: pd.DataFrame, test: pd.DataFrame, rul: np.ndarray) -> dict:
    life = train.groupby("unit")["cycle"].max()
    return {
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_units": int(train["unit"].nunique()),
        "n_test_units": int(test["unit"].nunique()),
        "life_min": int(life.min()),
        "life_max": int(life.max()),
        "life_mean": float(life.mean()),
        "rul_last_min": float(rul.min()),
        "rul_last_max": float(rul.max()),
        "rul_last_mean": float(rul.mean()),
    }
