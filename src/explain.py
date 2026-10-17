"""Explainability: permutation importance (+ SHAP if available)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_top(model, X, y, feat_names, n_repeats=5, seed=42, top_k=15,
                    scoring: str = "neg_root_mean_squared_error") -> pd.DataFrame:
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed,
                               n_jobs=-1, scoring=scoring)
    df = pd.DataFrame({"feature": feat_names, "importance": r.importances_mean,
                       "std": r.importances_std}).sort_values("importance", ascending=False)
    return df.head(top_k)


def try_shap_top(model, X, feat_names, top_k=15, nsample=200):
    try:
        import shap
        samp = X[:nsample] if len(X) > nsample else X
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(samp)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        imp = np.abs(sv).mean(axis=0)
        df = pd.DataFrame({"feature": feat_names, "shap_importance": imp}).sort_values(
            "shap_importance", ascending=False)
        return df.head(top_k)
    except Exception as e:
        return pd.DataFrame({"note": [f"SHAP unavailable: {e}"]})


def sensor_group_map(feat_names: list[str]) -> dict[str, list[str]]:
    """Map each engineered column to its base sensor (s11_rm10 -> s11)."""
    import re
    groups: dict[str, list[str]] = {}
    for f in feat_names:
        m = re.match(r"^(s\d+|cycle|op\d+)", f)
        key = m.group(1) if m else f
        groups.setdefault(key, []).append(f)
    return groups


def grouped_permutation_importance(model, X, y, feat_names: list[str],
                                   scoring: str = "neg_root_mean_squared_error",
                                   n_repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    """Permute all columns of one sensor jointly; drop in `scoring` vs baseline."""
    import numpy as np
    from sklearn.metrics import get_scorer
    Xa = np.asarray(X)
    ya = np.asarray(y)
    scorer = get_scorer(scoring)
    base = float(scorer(model, Xa, ya))
    rng = np.random.RandomState(seed)
    rows = []
    for gname, cols in sensor_group_map(list(feat_names)).items():
        idx = [list(feat_names).index(c) for c in cols]
        drops = []
        for _ in range(n_repeats):
            Xp = Xa.copy()
            Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
            drops.append(base - float(scorer(model, Xp, ya)))
        rows.append({"sensor": gname, "importance": float(np.mean(drops)),
                     "std": float(np.std(drops)), "n_features": len(idx)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def shap_beeswarm(model, X, feat_names: list[str], out_path: str, nsample: int = 500,
                  max_display: int = 15) -> bool:
    """SHAP beeswarm summary for a tree model; returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        Xa = np.asarray(X)
        Xs = Xa[:nsample] if len(Xa) > nsample else Xa
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(Xs)
        if isinstance(sv, list):
            sv = np.asarray(sv)[0]
        shap.summary_plot(np.asarray(sv), Xs, feature_names=list(feat_names),
                          max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False
