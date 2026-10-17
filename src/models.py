"""Model definitions: sklearn/boosting + PyTorch sequence models."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:  # pragma: no cover
    TORCH_OK = False
    torch = None  # type: ignore
    nn = None  # type: ignore


def get_regressors(seed: int = 42, fast: bool = False) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    n = 100 if fast else 300
    models = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=n, max_depth=None, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1),
        "hgb": HistGradientBoostingRegressor(max_iter=n, learning_rate=0.08, max_depth=None,
                                             random_state=seed),
        "mlp": MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", max_iter=400,
                            early_stopping=True, n_iter_no_change=20, random_state=seed),
    }
    try:
        import xgboost as xgb
        models["xgb"] = xgb.XGBRegressor(n_estimators=n, max_depth=6, learning_rate=0.08,
                                        subsample=0.9, colsample_bytree=0.8,
                                        random_state=seed, n_jobs=-1, tree_method="hist")
    except Exception as e:
        log.warning("xgboost unavailable: %s", e)
    try:
        import lightgbm as lgb
        models["lgbm"] = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.08, num_leaves=63,
                                          random_state=seed, n_jobs=-1, verbose=-1)
    except Exception as e:
        log.warning("lightgbm unavailable: %s", e)
    return models


def get_classifiers(seed: int = 42, fast: bool = False) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    n = 100 if fast else 300
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=n, min_samples_leaf=2, class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(max_iter=n, learning_rate=0.08, class_weight="balanced",
                                              random_state=seed),
    }
    try:
        import xgboost as xgb
        models["xgb"] = xgb.XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.08,
                                         subsample=0.9, colsample_bytree=0.8,
                                         random_state=seed, n_jobs=-1, tree_method="hist",
                                         eval_metric="mlogloss")
    except Exception as e:
        log.warning("xgboost unavailable: %s", e)
    try:
        import lightgbm as lgb
        models["lgbm"] = lgb.LGBMClassifier(n_estimators=n, learning_rate=0.08, num_leaves=63,
                                           class_weight="balanced", random_state=seed,
                                           n_jobs=-1, verbose=-1)
    except Exception as e:
        log.warning("lightgbm unavailable: %s", e)
    return models


if TORCH_OK:
    class CNN1D(torch.nn.Module):  # type: ignore
        def __init__(self, n_feat: int):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(n_feat, 64, 5, padding=2), torch.nn.ReLU(),
                torch.nn.Conv1d(64, 64, 5, padding=2), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
                torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 1))

        def forward(self, x):
            return self.net(x.transpose(1, 2)).squeeze(-1)

    class LSTMReg(torch.nn.Module):  # type: ignore
        def __init__(self, n_feat: int, hidden: int = 64, layers: int = 2):
            super().__init__()
            self.lstm = torch.nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=0.2)
            self.head = torch.nn.Sequential(torch.nn.Linear(hidden, 64), torch.nn.ReLU(),
                                            torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(-1)
else:  # pragma: no cover
    CNN1D = None  # type: ignore
    LSTMReg = None  # type: ignore
