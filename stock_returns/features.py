"""Leakage-aware feature and next-session target construction."""

from __future__ import annotations

import pandas as pd


def build_features(data: pd.DataFrame, ticker: str, benchmark: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build features known at close and the following session's stock return.

    `data` must contain close-price columns named `<SYMBOL>_close` and a
    `<TICKER>_volume` column. The target is aligned to the feature date.
    """
    ticker = ticker.upper()
    benchmark = benchmark.upper()
    stock_close = data[f"{ticker}_close"].astype(float)
    benchmark_close = data[f"{benchmark}_close"].astype(float)
    volume = data[f"{ticker}_volume"].astype(float)
    stock_returns = stock_close.pct_change(fill_method=None)
    benchmark_returns = benchmark_close.pct_change(fill_method=None)

    features = pd.DataFrame(index=data.index)
    for window in (1, 5, 20):
        features[f"stock_return_{window}d"] = stock_close.pct_change(window, fill_method=None)

    for window in (5, 20, 50):
        moving_average = stock_close.rolling(window=window, min_periods=window).mean()
        features[f"price_to_ma_{window}d"] = stock_close / moving_average - 1

    for window in (5, 20):
        features[f"volatility_{window}d"] = stock_returns.rolling(
            window=window, min_periods=window
        ).std()
        features[f"relative_return_{window}d"] = (
            stock_close.pct_change(window, fill_method=None)
            - benchmark_close.pct_change(window, fill_method=None)
        )

    features["volume_change_1d"] = volume.pct_change()
    features["volume_to_ma_20d"] = volume / volume.rolling(window=20, min_periods=20).mean() - 1

    target = stock_returns.shift(-1).rename("next_day_return")
    valid = features.notna().all(axis=1) & target.notna()
    return features.loc[valid], target.loc[valid]
