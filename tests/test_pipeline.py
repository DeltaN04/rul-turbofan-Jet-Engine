"""Pytest suite: PHM score, label boundaries, group-fold integrity, shapes."""
import numpy as np
import pandas as pd


def test_phm_score_hand_computed():
    from src.evaluate import phm_score
    # d = pred-true: [-10, +10] -> exp(10/13)-1 + exp(10/10)-1
    s = phm_score(np.array([100.0, 100.0]), np.array([90.0, 110.0]))
    expected = (np.exp(10 / 13) - 1) + (np.exp(10 / 10) - 1)
    assert abs(s - expected) < 1e-9
    # late (positive d) penalised more than early of same magnitude
    late = phm_score(np.array([100.0]), np.array([110.0]))
    early = phm_score(np.array([100.0]), np.array([90.0]))
    assert late > early


def test_class_boundaries():
    from src.data import add_class_labels
    df = pd.DataFrame({"RUL": [51.0, 50.0, 30.0, 20.0, 19.0, 0.0]})
    out = add_class_labels(df)
    assert out["cls3"].tolist() == ["safe", "warning", "warning", "warning", "critical", "critical"]
    assert out["bin30"].tolist() == [0, 0, 1, 1, 1, 1]


def test_no_unit_overlap_folds():
    from sklearn.model_selection import GroupKFold
    rng = np.random.RandomState(0)
    units = np.repeat(np.arange(20), 10)
    X = rng.randn(200, 3)
    y = rng.randn(200)
    for tr, va in GroupKFold(3).split(X, y, units):
        assert len(set(units[tr]) & set(units[va])) == 0


def test_feature_shapes():
    from src.data import load_split, add_rul_train, add_piecewise_rul
    from src.preprocess import ConditionNormalizer, feature_matrix
    tr, _, _ = load_split("data/raw", "FD001")
    tr = add_piecewise_rul(add_rul_train(tr), 125)
    keep = [c for c in [f"s{i}" for i in range(1, 22)] if tr[c].var() > 1e-6]
    norm = ConditionNormalizer(mode="global").fit(tr, keep)
    trn = norm.transform(tr)
    X, cols = feature_matrix(trn, keep, [10, 30])
    assert X.shape[0] == len(trn) and X.shape[1] == len(cols) and len(cols) > len(keep)


def test_fd002_condition_proxies_dropped():
    """A1: after normalise-first, s1/s5 (constant within each condition) must be
    dropped or have ~zero post-normalisation variance, and must not dominate
    permutation importance."""
    from src.data import load_split, add_rul_train, add_piecewise_rul
    from src.preprocess import ConditionNormalizer
    from src.data import SENSORS
    tr, _, _ = load_split("data/raw", "FD002")
    tr = add_piecewise_rul(add_rul_train(tr), 125)
    norm = ConditionNormalizer(mode="cluster", k=6, random_state=42).fit(tr, list(SENSORS))
    trn = norm.transform(tr)
    from src.preprocess import select_sensors_by_variance
    keep, drop = select_sensors_by_variance(trn, list(SENSORS), 1e-6)
    assert "s1" in drop and "s5" in drop, f"s1/s5 must drop after normalisation, dropped={drop}"
    assert float(trn["s1"].var()) < 1e-9 and float(trn["s5"].var()) < 1e-9


def test_no_future_leak_features():
    """cycle_norm (cycle/max_cycle) encodes failure time -> must not be a feature."""
    from src.data import load_split, add_rul_train, add_piecewise_rul
    from src.preprocess import ConditionNormalizer, feature_matrix
    tr, _, _ = load_split("data/raw", "FD001")
    tr = add_piecewise_rul(add_rul_train(tr), 125)
    keep = [c for c in [f"s{i}" for i in range(1, 22)] if tr[c].var() > 1e-6]
    norm = ConditionNormalizer(mode="global").fit(tr, keep)
    _, cols = feature_matrix(norm.transform(tr), keep, [10, 30])
    assert "cycle_norm" not in cols
    assert not any("max" in c for c in cols)


def test_sequences_padding():
    from src.preprocess import make_sequences
    df = pd.DataFrame({"unit": [1]*5, "cycle": [1,2,3,4,5],
                       "a": np.arange(5, dtype=float), "RUL_cap": np.arange(5, dtype=float)[::-1]})
    X, y, u = make_sequences(df, ["a"], window=30)
    assert X.shape == (5, 30, 1)
    assert y.tolist() == [4,3,2,1,0]


def test_no_duplicate_classification_rows():
    import glob
    import pandas as pd
    files = glob.glob("results/FD00*_classification.csv")
    assert files, "run the pipeline first"
    for f in files:
        df = pd.read_csv(f)
        assert not df.duplicated(subset=["fd", "task", "approach", "model", "class"]).any(), f
        # required columns incl. precision/F1/support (A5)
        for c in ["precision", "recall", "f1", "support", "accuracy", "balanced_acc",
                  "mcc", "kappa", "roc_auc", "pr_auc"]:
            assert c in df.columns, (f, c)


def test_no_unit_overlap_five_fold():
    from sklearn.model_selection import GroupKFold
    import numpy as np
    rng = np.random.RandomState(1)
    units = np.repeat(np.arange(25), 8)
    X = rng.randn(200, 3)
    y = rng.randn(200)
    assert GroupKFold(5).get_n_splits(X, y, units) == 5
    for tr, va in GroupKFold(5).split(X, y, units):
        assert len(set(units[tr]) & set(units[va])) == 0


def test_conformal_written_once():
    import glob
    for f in glob.glob("results/FD00*_conformal.txt"):
        lines = [ln for ln in open(f) if ln.strip()]
        assert len(lines) == 2, f"{f}: expected 2 lines (90/95), got {len(lines)}"


def test_capped_le_uncapped_when_many_above_cap():
    import pandas as pd
    df = pd.read_csv("results/master_regression.csv")
    assert "test_last_rmse_cap" in df.columns
    # FD002/FD004 have many true RUL > cap: capped RMSE must be well below uncapped
    for fd in ["FD002", "FD004"]:
        sub = df[(df.fd == fd) & (df.model == "hgb")]
        assert float(sub["test_last_rmse_cap"].iloc[0]) < float(sub["test_last_rmse_true"].iloc[0])
