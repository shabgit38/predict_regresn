"""Market data download and normalization."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def _extract_column(data: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """Extract one field from yfinance's single- or multi-ticker column layouts."""
    if isinstance(data.columns, pd.MultiIndex):
        for key in ((ticker, field), (field, ticker)):
            if key in data.columns:
                return data[key]
        raise ValueError(f"{field!r} data for {ticker!r} was not returned by Yahoo Finance.")

    if field not in data.columns:
        raise ValueError(f"{field!r} data for {ticker!r} was not returned by Yahoo Finance.")
    return data[field]


def download_market_data(ticker: str, benchmark: str, period: str = "10y") -> pd.DataFrame:
    """Download adjusted daily close and volume for a stock and its benchmark."""
    symbols = list(dict.fromkeys((ticker.upper(), benchmark.upper())))
    downloaded = yf.download(
        symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if downloaded.empty:
        raise ValueError("Yahoo Finance returned no market data for the requested symbols.")

    data = pd.DataFrame(index=downloaded.index)
    for symbol in symbols:
        data[f"{symbol}_close"] = _extract_column(downloaded, "Close", symbol)
    data[f"{ticker.upper()}_volume"] = _extract_column(downloaded, "Volume", ticker.upper())

    data = data.sort_index()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data
