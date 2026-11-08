# Stock Return Prediction

Starter project for predicting the next trading day's stock return from historical market data with an XGBoost regressor.

## What it does

- Downloads daily close prices and volume from the Kite Connect API.
- Builds lagged return, moving-average distance, rolling volatility, volume, and benchmark-relative features.
- Predicts the next trading day's close-to-close return.
- Uses a chronological train/test split and reports out-of-sample regression and directional metrics.

This is an educational modeling example, not investment advice. Historical performance does not guarantee future results.

## Quick start

Use Python 3.10 or later:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:KITE_API_KEY = "your_api_key"
$env:KITE_ACCESS_TOKEN = "your_access_token"
python -m stock_returns.cli --ticker RELIANCE --benchmark "NIFTY 50" --exchange NSE --period 10y
```

If PowerShell prevents environment activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and then `.venv\Scripts\python.exe -m stock_returns.cli ...`.

Set `KITE_API_KEY` and `KITE_ACCESS_TOKEN` from your Kite Connect app/session before running the command. Do not commit credentials. Kite symbols and benchmarks must exist on the selected exchange; use `--ticker`, `--benchmark`, and `--exchange` to select them. The benchmark is used to calculate relative-strength features; it is not itself a prediction target. `--period` accepts durations such as `10y`, `6mo`, and `90d`.

## Output

The command prints the selected feature names, the chronological test-period boundaries, MAE, RMSE, directional accuracy, and the most influential features from the fitted XGBoost model.

Kite historical candles provide close prices, not split/dividend-adjusted closes. Each feature row is computed from information available at that row's market close. The target is the following trading session's return. The final row, which has no known next-session return, is excluded. Rows with incomplete rolling windows are also excluded.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests use synthetic data and do not require a network connection.

## Notes

- Kite Connect historical data access requires an active API session and is subject to Zerodha's API limits and terms.
- The default split reserves the most recent 20% of observations for testing; it does not shuffle dates.
- For a realistic research workflow, add walk-forward validation, transaction costs, and point-in-time data checks before drawing conclusions.
