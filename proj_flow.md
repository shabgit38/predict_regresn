# Project Flow

## Purpose

This project predicts the next trading day's close-to-close stock return using daily market data and an XGBoost regression baseline.

## Implementation flow

1. **Authenticate with Kite Connect**
   - `stock_returns/auth.py` loads project credentials from `.env`.
   - `--login` opens the Kite login page, accepts a redirect URL or request token, exchanges it for an access token, and saves the token to `.env`.
   - Kite access tokens may expire and must be renewed through the login flow.

2. **Select inputs and download data**
   - `stock_returns/cli.py` accepts the ticker, benchmark, exchange, and history period.
   - `stock_returns/data.py` resolves the stock and benchmark symbols to Kite instrument tokens.
   - It downloads daily close prices and stock volume, normalizes timezone information, sorts observations chronologically, and aligns both symbols by date.

3. **Build leakage-aware features**
   - `stock_returns/features.py` computes stock returns over 1, 5, and 20 trading days.
   - It creates 5-, 20-, and 50-day moving-average distances, 5- and 20-day volatility and relative-return features, and 1-day/20-day volume features.
   - Features use information available at each market close.
   - The target is the next trading day's stock return.
   - The final row and rows with incomplete rolling windows are removed.

4. **Fit and evaluate the model**
   - `stock_returns/model.py` uses an XGBoost `XGBRegressor` baseline.
   - It performs a chronological 80/20 train/test split without shuffling observations.
   - It reports MAE, RMSE, directional accuracy, test-period boundaries, and ranked feature importances.

5. **Present command-line results**
   - `stock_returns/cli.py` runs the complete pipeline and prints the selected ticker, benchmark, evaluation period, metrics, and top features.

## Current project structure

- `stock_returns/auth.py`: Kite credentials and interactive login
- `stock_returns/data.py`: period parsing, instrument lookup, and historical data download
- `stock_returns/features.py`: feature and target construction
- `stock_returns/model.py`: chronological split, XGBoost fitting, and evaluation
- `stock_returns/cli.py`: command-line orchestration and output
- `tests/`: synthetic unit tests for authentication, data download, and feature construction

## Validation

The test suite uses `python -m unittest discover -s tests -v`. Tests cover request-token parsing and credential storage, instrument and token error handling, data normalization, next-session target alignment, removal of incomplete rows, and prevention of final-row leakage.

## Known limitations and next steps

- The baseline uses a single chronological holdout rather than walk-forward validation.
- Kite candle closes are not split/dividend-adjusted.
- API access is subject to Kite credentials and usage limits.
- Future work could add walk-forward evaluation, benchmark baselines, transaction costs, feature ablation, and point-in-time data checks.
