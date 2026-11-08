from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_returns.features import build_features


class BuildFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=100)
        self.data = pd.DataFrame(
            {
                "ABC_close": np.linspace(100, 150, len(dates)),
                "SPY_close": np.linspace(200, 260, len(dates)),
                "ABC_volume": np.linspace(1_000_000, 1_500_000, len(dates)),
            },
            index=dates,
        )

    def test_features_use_information_at_feature_date_and_target_is_next_return(self) -> None:
        features, target = build_features(self.data, "ABC", "SPY")

        first_date = features.index[0]
        original_position = self.data.index.get_loc(first_date)
        expected_target = (
            self.data["ABC_close"].iloc[original_position + 1]
            / self.data["ABC_close"].iloc[original_position]
            - 1
        )
        self.assertAlmostEqual(target.loc[first_date], expected_target)
        self.assertAlmostEqual(
            features.loc[first_date, "stock_return_1d"],
            self.data["ABC_close"].pct_change().loc[first_date],
        )

    def test_last_row_is_removed_because_it_has_no_next_day_target(self) -> None:
        features, target = build_features(self.data, "ABC", "SPY")

        self.assertEqual(features.index[-1], self.data.index[-2])
        self.assertTrue(features.index.equals(target.index))

    def test_features_skip_incomplete_rolling_windows(self) -> None:
        features, _ = build_features(self.data, "ABC", "SPY")

        self.assertGreaterEqual(self.data.index.get_loc(features.index[0]), 49)
        self.assertTrue(features.notna().all().all())


if __name__ == "__main__":
    unittest.main()
