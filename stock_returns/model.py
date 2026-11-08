"""Chronological model fitting and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


@dataclass(frozen=True)
class Evaluation:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    mae: float
    rmse: float
    directional_accuracy: float
    feature_importance: pd.Series


def fit_and_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    test_fraction: float = 0.2,
) -> Evaluation:
    """Fit on the earlier observations and evaluate on the latest observations."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not features.index.equals(target.index):
        raise ValueError("Feature and target indexes must match exactly.")

    split_at = int(len(features) * (1 - test_fraction))
    if split_at < 1 or split_at >= len(features):
        raise ValueError("Not enough rows for the requested chronological train/test split.")

    x_train, x_test = features.iloc[:split_at], features.iloc[split_at:]
    y_train, y_test = target.iloc[:split_at], target.iloc[split_at:]
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    importances = pd.Series(model.feature_importances_, index=features.columns)
    importances = importances.sort_values(ascending=False)
    return Evaluation(
        train_end=features.index[split_at - 1],
        test_start=features.index[split_at],
        test_end=features.index[-1],
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        directional_accuracy=float(np.mean(np.sign(predictions) == np.sign(y_test.to_numpy()))),
        feature_importance=importances,
    )
