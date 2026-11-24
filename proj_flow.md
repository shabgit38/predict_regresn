# Project Flow

## Purpose

This project predicts the next trading day's close-to-close stock return using daily market data and an XGBoost regression baseline.

## Implementation flow

1. **Authenticate with Kite Connect**
   - `stock_returns/auth.py` loads project credentials from `.env`.
   - The app's **Login to Kite** button opens the Kite login page. Kite redirects to the local app, which exchanges the request token and saves the access token to `.env`.
   - Kite access tokens may expire and must be renewed through the login flow.

2. **Select inputs and download data**
   - Root-level `app.py` provides a local browser form for ticker, benchmark, exchange, and history duration. Start it with `python -m streamlit run app.py --server.address 127.0.0.1`.
   - Input hints show ticker examples (`RELIANCE`, `INFY`), benchmark examples (`NIFTY 50`, `SENSEX`), and duration formats (`5y`, `6mo`, `90d`).
   - The sidebar provides a single Kite login button; the Kite app's Redirect URL must point to `http://127.0.0.1:8501/` for automatic local callback without manually pasting a token.
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

5. **Present results**
   - Root-level `app.py` displays the selected symbols, evaluation period, metrics, test observation count, and a feature-importance chart/table.
   - `stock_returns/cli.py` remains available as an optional terminal interface.

## Current project structure

- `stock_returns/auth.py`: Kite credentials and interactive login
- `stock_returns/data.py`: period parsing, instrument lookup, and historical data download
- `stock_returns/features.py`: feature and target construction
- `stock_returns/model.py`: chronological split, XGBoost fitting, and evaluation
- `app.py`: local browser interface, Kite login, and model results
- `stock_returns/cli.py`: optional command-line interface
- `tests/`: unit tests for authentication, UI rendering, data download, and feature construction

## Validation

The test suite uses `python -m unittest discover -s tests -v`. Tests cover request-token parsing and credential storage, instrument and token error handling, data normalization, next-session target alignment, removal of incomplete rows, and prevention of final-row leakage.

## Known limitations and next steps

- The baseline uses a single chronological holdout rather than walk-forward validation.
- Kite candle closes are not split/dividend-adjusted.
- API access is subject to Kite credentials and usage limits.
- Future work could add walk-forward evaluation, benchmark baselines, transaction costs, feature ablation, and point-in-time data checks.
