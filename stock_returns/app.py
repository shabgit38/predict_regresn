"""Local browser interface for stock return model evaluation."""

from __future__ import annotations

import os
import re

import streamlit as st
from kiteconnect.exceptions import KiteException
from requests.exceptions import RequestException

from stock_returns.auth import (
    exchange_kite_request_token,
    get_kite_login_url,
    load_kite_environment,
)
from stock_returns.data import download_market_data
from stock_returns.features import build_features
from stock_returns.model import fit_and_evaluate

st.set_page_config(
    page_title="Stock Return Research",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-top: 2rem; }
    [data-testid="stMetric"] {
        background: #f5f7fa;
        border: 1px solid #e5e9f0;
        padding: 1rem;
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

load_kite_environment()

st.title("Stock Return Research")
st.caption(
    "Configure a stock, benchmark, and history window to evaluate the next-session "
    "return model. Historical results are not investment advice."
)

with st.sidebar:
    st.subheader("Kite Connect")
    if os.environ.get("KITE_ACCESS_TOKEN"):
        st.success("Access token is configured.")
    else:
        st.info("Connect your Kite account to download market data.")

    try:
        login_url = get_kite_login_url()
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.link_button("Open Kite login", login_url, use_container_width=True)
        request_token = st.text_input(
            "Redirect URL or request token",
            placeholder="Paste the URL containing request_token=... or the token",
            type="password",
        )
        if st.button("Connect to Kite", use_container_width=True):
            if not request_token.strip():
                st.error("Paste the Kite redirect URL or request token first.")
            else:
                try:
                    exchange_kite_request_token(request_token)
                except (KiteException, OSError, RequestException, ValueError) as exc:
                    st.error(f"Kite login failed: {exc}")
                else:
                    st.success("Kite access token saved locally.")
                    st.rerun()

st.subheader("Model inputs")
with st.form("model_inputs"):
    col_ticker, col_benchmark = st.columns(2)
    with col_ticker:
        ticker = st.text_input(
            "Ticker",
            placeholder="For example: RELIANCE or INFY",
            help="Enter the Kite trading symbol for the stock.",
        )
    with col_benchmark:
        benchmark = st.text_input(
            "Benchmark",
            placeholder="For example: NIFTY 50",
            help="Enter the Kite trading symbol to use for relative-return features.",
        )

    col_exchange, col_period = st.columns(2)
    with col_exchange:
        exchange = st.text_input(
            "Exchange",
            value="NSE",
            help="Kite exchange code, for example NSE or BSE.",
        )
    with col_period:
        period = st.text_input(
            "History duration",
            value="5y",
            help="Format: number followed by d (days), mo (months), or y (years), e.g. 90d, 6mo, 5y.",
        )
    submitted = st.form_submit_button("Run evaluation", type="primary", use_container_width=True)

if submitted:
    ticker = ticker.strip().upper()
    benchmark = benchmark.strip().upper()
    exchange = exchange.strip().upper()
    period = period.strip().lower()

    if not ticker or not benchmark or not exchange:
        st.error("Ticker, benchmark, and exchange are required.")
    elif not re.fullmatch(r"[1-9]\d*(?:d|mo|y)", period):
        st.error("Duration must use the format <number><unit>, for example 90d, 6mo, or 5y.")
    else:
        try:
            with st.spinner("Downloading market data and evaluating the model..."):
                data = download_market_data(ticker, benchmark, period, exchange)
                features, target = build_features(data, ticker, benchmark)
                result = fit_and_evaluate(features, target)
        except (KiteException, OSError, RequestException, RuntimeError, ValueError) as exc:
            st.error(f"Could not complete the evaluation: {exc}")
        else:
            st.subheader(f"{ticker} vs {benchmark}")
            st.caption(
                f"Training data through {result.train_end.date()} · "
                f"Test period {result.test_start.date()} to {result.test_end.date()}"
            )

            test_observations = len(features.loc[result.test_start :])
            metric_columns = st.columns(4)
            metric_columns[0].metric("MAE", f"{result.mae:.6f}")
            metric_columns[1].metric("RMSE", f"{result.rmse:.6f}")
            metric_columns[2].metric("Directional accuracy", f"{result.directional_accuracy:.1%}")
            metric_columns[3].metric("Test observations", f"{test_observations:,}")

            st.subheader("Top feature importances")
            importance = (
                result.feature_importance.head(10)
                .rename_axis("Feature")
                .rename("Importance")
                .reset_index()
            )
            st.bar_chart(importance.set_index("Feature"))
            st.dataframe(importance, hide_index=True, use_container_width=True)
