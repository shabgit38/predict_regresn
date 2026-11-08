"""Market data download and normalization."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import pandas as pd
from kiteconnect import KiteConnect


def _period_start(period: str, end: datetime) -> datetime:
    """Convert a duration such as 10y or 6mo into its start date."""
    match = re.fullmatch(r"(\d+)(d|mo|y)", period.lower())
    if match is None:
        raise ValueError("Period must be a positive number followed by d, mo, or y (for example 10y).")
    amount = int(match.group(1))
    if amount < 1:
        raise ValueError("Period must be greater than zero.")
    unit = match.group(2)
    if unit == "d":
        return end - timedelta(days=amount)
    months = amount * (12 if unit == "y" else 1)
    return end - pd.DateOffset(months=months)


def download_market_data(
    ticker: str,
    benchmark: str,
    period: str = "10y",
    exchange: str = "NSE",
) -> pd.DataFrame:
    """Download daily close and volume candles for a stock and its benchmark from Kite."""
    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        raise ValueError("Set the KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.")

    client = KiteConnect(api_key=api_key)
    client.set_access_token(access_token)
    exchange = exchange.upper()
    symbols = list(dict.fromkeys((ticker.upper(), benchmark.upper())))
    instruments = client.instruments(exchange)
    tokens: dict[str, int] = {}
    for symbol in symbols:
        match = next(
            (
                instrument
                for instrument in instruments
                if instrument["tradingsymbol"].upper() == symbol
                and instrument["exchange"].upper() == exchange
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Instrument {symbol!r} was not found on the {exchange} exchange.")
        tokens[symbol] = int(match["instrument_token"])

    end = datetime.now()
    start = _period_start(period, end)
    data = pd.DataFrame()
    for symbol in symbols:
        candles = client.historical_data(tokens[symbol], start, end, "day")
        if not candles:
            raise ValueError(f"Kite Connect returned no historical data for {symbol!r}.")
        frame = pd.DataFrame(candles)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        data[f"{symbol}_close"] = frame["close"]
        if symbol == ticker.upper():
            data[f"{symbol}_volume"] = frame["volume"]

    data.index = pd.DatetimeIndex(data.index).tz_localize(None)
    return data.sort_index()
