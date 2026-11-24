"""Command-line entry point for the stock return prediction baseline."""

from __future__ import annotations

import argparse
import re

from stock_returns.auth import login_to_kite
from stock_returns.data import download_market_data
from stock_returns.features import build_features
from stock_returns.model import fit_and_evaluate


def _prompt_required(label: str, hint: str) -> str:
    while True:
        value = input(f"{label} ({hint}): ").strip()
        if value:
            return value
        print(f"{label} is required.")


def _prompt_period() -> str:
    while True:
        value = input("Duration (format: 5y, 6mo, or 90d) [5y]: ").strip().lower()
        if not value:
            return "5y"
        if re.fullmatch(r"[1-9]\d*(?:d|mo|y)", value):
            return value
        print("Duration must use format <number><unit>, such as 5y, 6mo, or 90d.")


def _get_user_inputs(args: argparse.Namespace) -> None:
    if not args.ticker:
        args.ticker = _prompt_required(
            "Ticker",
            "stock symbol, for example RELIANCE or INFY",
        )
    if not args.benchmark:
        args.benchmark = _prompt_required(
            "Benchmark",
            "benchmark symbol, for example NIFTY 50 or SENSEX",
        )
    if not args.period:
        args.period = _prompt_period()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next trading day's stock return using Kite Connect data.")
    parser.add_argument("--ticker", help="Stock trading symbol, for example RELIANCE or INFY.")
    parser.add_argument("--benchmark", help="Benchmark trading symbol, for example NIFTY 50 or SENSEX.")
    parser.add_argument("--exchange", default="NSE", help="Kite exchange (default: NSE).")
    parser.add_argument("--period", help="History duration, such as 5y, 6mo, or 90d.")
    parser.add_argument("--login", action="store_true", help="Log in to Kite and save a new access token.")
    args = parser.parse_args()

    if args.login:
        login_to_kite()
        return

    _get_user_inputs(args)
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
