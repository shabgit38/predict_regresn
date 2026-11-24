from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from stock_returns.cli import _get_user_inputs, _prompt_period, _prompt_required


class UserInputTests(unittest.TestCase):
    @patch("builtins.input", return_value="  RELIANCE  ")
    def test_required_prompt_trims_input(self, input_prompt: unittest.mock.Mock) -> None:
        self.assertEqual(_prompt_required("Ticker", "stock symbol"), "RELIANCE")

    @patch("builtins.input", return_value="")
    def test_period_prompt_accepts_default(self, input_prompt: unittest.mock.Mock) -> None:
        self.assertEqual(_prompt_period(), "5y")

    @patch("builtins.input", side_effect=["invalid", "6mo"])
    def test_period_prompt_retries_invalid_format(self, input_prompt: unittest.mock.Mock) -> None:
        self.assertEqual(_prompt_period(), "6mo")

    @patch("stock_returns.cli._prompt_period", return_value="5y")
    @patch("stock_returns.cli._prompt_required", side_effect=["INFY", "NIFTY 50"])
    def test_missing_inputs_are_prompted(
        self,
        prompt_required: unittest.mock.Mock,
        prompt_period: unittest.mock.Mock,
    ) -> None:
        args = argparse.Namespace(ticker=None, benchmark=None, period=None)

        _get_user_inputs(args)

        self.assertEqual(args.ticker, "INFY")
        self.assertEqual(args.benchmark, "NIFTY 50")
        self.assertEqual(args.period, "5y")

    def test_supplied_inputs_are_not_prompted(self) -> None:
        args = argparse.Namespace(ticker="TCS", benchmark="SENSEX", period="90d")

        with patch("stock_returns.cli._prompt_required") as prompt_required:
            with patch("stock_returns.cli._prompt_period") as prompt_period:
                _get_user_inputs(args)

        prompt_required.assert_not_called()
        prompt_period.assert_not_called()


if __name__ == "__main__":
    unittest.main()
