"""Command-line entry point for the stock return prediction baseline."""

from __future__ import annotations

import argparse

from stock_returns.auth import login_to_kite
from stock_returns.data import download_market_data
from stock_returns.features import build_features
from stock_returns.model import fit_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next trading day's stock return using Kite Connect data.")
    parser.add_argument("--ticker", default="RELIANCE", help="Stock trading symbol (default: RELIANCE).")
    parser.add_argument("--benchmark", default="NIFTY 50", help="Benchmark trading symbol (default: NIFTY 50).")
    parser.add_argument("--exchange", default="NSE", help="Kite exchange (default: NSE).")
    parser.add_argument("--period", default="10y", help="History duration, such as 10y, 6mo, or 90d (default: 10y).")
    parser.add_argument("--login", action="store_true", help="Log in to Kite and save a new access token.")
    args = parser.parse_args()

    if args.login:
        login_to_kite()
        return

    data = download_market_data(args.ticker, args.benchmark, args.period, args.exchange)
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
