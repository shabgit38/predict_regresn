"""Command-line entry point for the stock return prediction baseline."""

from __future__ import annotations

import argparse

from stock_returns.data import download_market_data
from stock_returns.features import build_features
from stock_returns.model import fit_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next trading day's stock return.")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL).")
    parser.add_argument("--benchmark", default="SPY", help="Market benchmark (default: SPY).")
    parser.add_argument("--period", default="10y", help="Yahoo Finance history period (default: 10y).")
    args = parser.parse_args()

    data = download_market_data(args.ticker, args.benchmark, args.period)
    features, target = build_features(data, args.ticker, args.benchmark)
    result = fit_and_evaluate(features, target)

    print(f"Ticker: {args.ticker.upper()} | Benchmark: {args.benchmark.upper()}")
    print(f"Train through: {result.train_end.date()}")
    print(f"Test period: {result.test_start.date()} to {result.test_end.date()}")
    print(f"Test observations: {len(features.loc[result.test_start:])}")
    print(f"MAE: {result.mae:.6f}")
    print(f"RMSE: {result.rmse:.6f}")
    print(f"Directional accuracy: {result.directional_accuracy:.1%}")
    print("Top feature importances:")
    print(result.feature_importance.head(10).to_string())


if __name__ == "__main__":
    main()
