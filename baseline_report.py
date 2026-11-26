"""Baseline comparisons and uncertainty summaries for model evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

_BOOTSTRAP_REPLICATES = 2000
_BLOCK_SIZE = 5
_RANDOM_STATE = 42


@dataclass(frozen=True)
class BaselineReport:
    model_mae: float
    model_rmse: float
    zero_mae: float
    zero_rmse: float
    train_mean_mae: float
    train_mean_rmse: float
    mae_gap_pct: float | None
    mae_gap_ci_low: float | None
    mae_gap_ci_high: float | None
    directional_hits: int
    observations: int
    directional_accuracy: float
    directional_ci_low: float
    directional_ci_high: float
    directional_p_value: float
    always_up_rate: float
    verdict: str


def _as_finite_vector(values: object, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional sequence.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values.")
    return array


def _block_bootstrap_mae_gap(
    model_errors: np.ndarray,
    zero_errors: np.ndarray,
    zero_mae: float,
) -> tuple[float, float]:
    size = len(model_errors)
    block_count = ceil(size / _BLOCK_SIZE)
    rng = np.random.default_rng(_RANDOM_STATE)
    bootstrap_gaps = np.empty(_BOOTSTRAP_REPLICATES)
    offsets = np.arange(_BLOCK_SIZE)

    for replicate in range(_BOOTSTRAP_REPLICATES):
        starts = rng.integers(0, size, size=block_count)
        indices = ((starts[:, None] + offsets) % size).ravel()[:size]
        model_mae = model_errors[indices].mean()
        zero_baseline_mae = zero_errors[indices].mean()
        bootstrap_gaps[replicate] = 100 * (model_mae - zero_baseline_mae) / zero_mae

    lower, upper = np.percentile(bootstrap_gaps, [2.5, 97.5])
    return float(lower), float(upper)


def build_report(y_train: object, y_test: object, pred: object) -> BaselineReport:
    """Compare model errors with simple baselines on the same chronological test set."""
    from scipy.stats import binomtest

    train = _as_finite_vector(y_train, "y_train")
    actual = _as_finite_vector(y_test, "y_test")
    predictions = _as_finite_vector(pred, "pred")
    if actual.size != predictions.size:
        raise ValueError("y_test and pred must contain the same number of observations.")

    train_mean_predictions = np.full(actual.shape, train.mean())
    zero_errors = np.abs(actual)
    model_errors = np.abs(actual - predictions)
    zero_mae = float(zero_errors.mean())
    model_mae = float(model_errors.mean())
    model_rmse = float(np.sqrt(np.mean(np.square(actual - predictions))))
    zero_rmse = float(np.sqrt(np.mean(np.square(actual))))
    train_mean_errors = actual - train_mean_predictions
    train_mean_mae = float(np.mean(np.abs(train_mean_errors)))
    train_mean_rmse = float(np.sqrt(np.mean(np.square(train_mean_errors))))

    mae_gap_pct = None
    mae_gap_ci_low = None
    mae_gap_ci_high = None
    if zero_mae > 0:
        mae_gap_pct = 100 * (model_mae - zero_mae) / zero_mae
        mae_gap_ci_low, mae_gap_ci_high = _block_bootstrap_mae_gap(
            model_errors,
            zero_errors,
            zero_mae,
        )

    directional_hits = int(np.count_nonzero(np.sign(predictions) == np.sign(actual)))
    observations = int(actual.size)
    direction_test = binomtest(directional_hits, observations, p=0.5)
    direction_ci = direction_test.proportion_ci(confidence_level=0.95, method="exact")
    directional_accuracy = directional_hits / observations
    always_up_rate = float(np.mean(actual > 0))

    mae_evidence = mae_gap_ci_high is not None and mae_gap_ci_high < 0
    direction_evidence = direction_ci.low > 0.5
    if mae_evidence and direction_evidence:
        verdict = "Evidence of skill"
    elif mae_evidence or direction_evidence:
        verdict = "Partial evidence"
    else:
        verdict = "No evidence yet"

    return BaselineReport(
        model_mae=model_mae,
        model_rmse=model_rmse,
        zero_mae=zero_mae,
        zero_rmse=zero_rmse,
        train_mean_mae=train_mean_mae,
        train_mean_rmse=train_mean_rmse,
        mae_gap_pct=mae_gap_pct,
        mae_gap_ci_low=mae_gap_ci_low,
        mae_gap_ci_high=mae_gap_ci_high,
        directional_hits=directional_hits,
        observations=observations,
        directional_accuracy=directional_accuracy,
        directional_ci_low=float(direction_ci.low),
        directional_ci_high=float(direction_ci.high),
        directional_p_value=float(direction_test.pvalue),
        always_up_rate=always_up_rate,
        verdict=verdict,
    )


def render_report(st: object, report: BaselineReport) -> None:
    """Render baseline and uncertainty results using Streamlit's UI API."""
    import pandas as pd

    st.subheader("Baseline comparison")
    comparison = pd.DataFrame(
        [
            ("Model", report.model_mae, report.model_rmse),
            ("Zero-return", report.zero_mae, report.zero_rmse),
            ("Train-mean", report.train_mean_mae, report.train_mean_rmse),
        ],
        columns=("Method", "MAE", "RMSE"),
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    if report.mae_gap_pct is None:
        st.caption("MAE gap vs zero-return is undefined because the zero-return baseline has zero MAE.")
    else:
        st.metric(
            "MAE gap vs zero-return",
            f"{report.mae_gap_pct:+.2f}%",
            help="Model MAE minus zero-return MAE, relative to zero-return MAE. Negative is better.",
        )
        if report.mae_gap_ci_low is None or report.mae_gap_ci_high is None:
            raise ValueError("MAE gap confidence interval is missing.")
        st.caption(
            "95% block-bootstrap CI: "
            f"[{report.mae_gap_ci_low:+.2f}%, {report.mae_gap_ci_high:+.2f}%]. "
            "If the interval crosses 0, the gap is not distinguishable from noise."
        )

    columns = st.columns(3)
    columns[0].metric(
        "Directional accuracy",
        f"{report.directional_accuracy:.1%}",
        help=f"{report.directional_hits} correct directions out of {report.observations} test days.",
    )
    columns[1].metric(
        "95% exact direction CI",
        f"{report.directional_ci_low:.1%}–{report.directional_ci_high:.1%}",
    )
    columns[2].metric(
        "P-value vs 50%",
        f"{report.directional_p_value:.4f}",
        help="Two-sided exact binomial test against 50% directional accuracy.",
    )
    st.caption(
        "Always-up baseline: the actual return was positive on "
        f"{report.always_up_rate:.1%} of test days."
    )

    if report.verdict == "Evidence of skill":
        st.success(report.verdict)
    elif report.verdict == "Partial evidence":
        st.warning(report.verdict)
    else:
        st.info(report.verdict)
