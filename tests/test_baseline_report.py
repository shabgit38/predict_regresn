from __future__ import annotations

import unittest

from baseline_report import build_report


class BaselineReportTests(unittest.TestCase):
    def test_build_report_compares_baselines_and_direction(self) -> None:
        report = build_report(
            y_train=[0.01, 0.02, 0.03],
            y_test=[0.01, -0.02, 0.04, -0.01],
            pred=[0.02, -0.01, -0.03, 0.01],
        )

        self.assertAlmostEqual(report.zero_mae, 0.02)
        self.assertAlmostEqual(report.train_mean_mae, 0.025)
        self.assertEqual(report.directional_hits, 2)
        self.assertEqual(report.observations, 4)
        self.assertAlmostEqual(report.directional_accuracy, 0.5)
        self.assertAlmostEqual(report.always_up_rate, 0.5)
        self.assertIsNotNone(report.mae_gap_ci_low)
        self.assertLessEqual(report.mae_gap_ci_low, report.mae_gap_ci_high)

    def test_zero_return_baseline_with_zero_error_has_no_relative_gap(self) -> None:
        report = build_report(y_train=[-0.01, 0.01], y_test=[0.0, 0.0], pred=[0.0, 0.01])

        self.assertEqual(report.zero_mae, 0.0)
        self.assertIsNone(report.mae_gap_pct)
        self.assertIsNone(report.mae_gap_ci_low)
        self.assertEqual(report.verdict, "No evidence yet")

    def test_verdict_requires_improved_mae_and_directional_evidence(self) -> None:
        actual = [0.01, -0.02] * 10
        train = [0.01, -0.01, 0.02]

        evidence = build_report(y_train=train, y_test=actual, pred=actual)
        direction_only = build_report(y_train=train, y_test=actual, pred=[value * 2 for value in actual])

        self.assertEqual(evidence.verdict, "Evidence of skill")
        self.assertEqual(direction_only.verdict, "Partial evidence")

    def test_rejects_mismatched_test_and_prediction_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number"):
            build_report(y_train=[0.01], y_test=[0.01, 0.02], pred=[0.01])


if __name__ == "__main__":
    unittest.main()
