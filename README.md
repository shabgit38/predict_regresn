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
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and set KITE_API_KEY and KITE_API_SECRET from your Kite Connect app.
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The app opens in your browser. In your Kite Connect app settings, set the **Redirect URL** to `http://127.0.0.1:8501/`. Click **Login to Kite** in the app sidebar and sign in; Kite redirects back to the app, which exchanges the request token automatically. Then enter:

- **Ticker:** Kite stock symbol, such as `RELIANCE` or `INFY`
- **Benchmark:** Kite symbol, such as `NIFTY 50` or `SENSEX`
- **Exchange:** Kite exchange code, such as `NSE` or `BSE`
- **History duration:** number plus unit, such as `5y`, `6mo`, or `90d`

The app binds to `127.0.0.1`, so it is only available on your computer. No environment activation is needed; use the `.venv\Scripts\python.exe` commands shown above.

The app loads your Kite API key and API secret from `.env`. The login callback saves the exchanged access token to `.env`. If the token expires, click **Login to Kite** again. Keep `.env` private and never commit it. Enter Kite stock and benchmark symbols that exist on the selected exchange. The benchmark is used for relative-strength features, not as the prediction target. Duration accepts values such as `5y`, `6mo`, and `90d`.

## Results

The app displays MAE, RMSE, directional accuracy, test-period boundaries, test observation count, and a chart/table of the most influential features from the fitted XGBoost model. These are historical holdout evaluation results, not a forecast of a specific future return.

Kite historical candles provide close prices, not split/dividend-adjusted closes. Each feature row is computed from information available at that row's market close. The target is the following trading session's return. The final row, which has no known next-session return, is excluded. Rows with incomplete rolling windows are also excluded.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic data and do not require a network connection.

## Notes

- Kite Connect historical data access requires an active API session and is subject to Zerodha's API limits and terms.
- The default split reserves the most recent 20% of observations for testing; it does not shuffle dates.
- For a realistic research workflow, add walk-forward validation, transaction costs, and point-in-time data checks before drawing conclusions.
