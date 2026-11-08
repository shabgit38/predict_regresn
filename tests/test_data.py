from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

from stock_returns.data import download_market_data


class DownloadMarketDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instruments = [
            {"tradingsymbol": "ABC", "exchange": "NSE", "instrument_token": 123},
            {"tradingsymbol": "NIFTY 50", "exchange": "NSE", "instrument_token": 456},
        ]
        self.candles = [
            {"date": pd.Timestamp("2025-01-02", tz="Asia/Kolkata"), "close": 100.0, "volume": 1000},
            {"date": pd.Timestamp("2025-01-03", tz="Asia/Kolkata"), "close": 101.0, "volume": 1100},
        ]

    @patch.dict(os.environ, {"KITE_API_KEY": "test-key", "KITE_ACCESS_TOKEN": "test-token"})
    @patch("stock_returns.data.KiteConnect")
    def test_downloads_daily_candles_for_resolved_instruments(self, kite_connect: unittest.mock.Mock) -> None:
        client = kite_connect.return_value
        client.instruments.return_value = self.instruments
        client.historical_data.return_value = self.candles

        data = download_market_data("abc", "nifty 50", period="10d")

        kite_connect.assert_called_once_with(api_key="test-key")
        client.set_access_token.assert_called_once_with("test-token")
        self.assertEqual(
            [call.args[0] for call in client.historical_data.call_args_list],
            [123, 456],
        )
        self.assertEqual(
            list(data.columns),
            ["ABC_close", "ABC_volume", "NIFTY 50_close"],
        )
        self.assertIsNone(data.index.tz)
        self.assertEqual(data["ABC_close"].tolist(), [100.0, 101.0])

    @patch.dict(os.environ, {}, clear=True)
    def test_requires_kite_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "KITE_API_KEY and KITE_ACCESS_TOKEN"):
            download_market_data("ABC", "NIFTY 50")

    @patch.dict(os.environ, {"KITE_API_KEY": "test-key", "KITE_ACCESS_TOKEN": "test-token"})
    @patch("stock_returns.data.KiteConnect")
    def test_raises_for_unknown_instrument(self, kite_connect: unittest.mock.Mock) -> None:
        kite_connect.return_value.instruments.return_value = self.instruments[:1]

        with self.assertRaisesRegex(ValueError, "NIFTY 50.*not found"):
            download_market_data("ABC", "NIFTY 50")


if __name__ == "__main__":
    unittest.main()
