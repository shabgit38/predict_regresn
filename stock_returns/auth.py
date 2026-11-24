"""Kite Connect credentials and interactive login."""

from __future__ import annotations

import getpass
import os
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

ENV_FILE = Path(".env")


def load_kite_environment() -> None:
    """Load project-local secrets without overriding existing environment variables."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _required_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} in the project .env file or environment.")
    return value


def _get_request_token(value: str) -> str:
    parsed = parse_qs(urlparse(value).query)
    tokens = parsed.get("request_token")
    if tokens and tokens[0]:
        return tokens[0]
    if "?" not in value and "&" not in value and "=" not in value and value:
        return value
    raise ValueError("The supplied value does not contain a Kite request_token.")


def get_kite_login_url() -> str:
    """Create the Kite login URL using the configured API key."""
    load_kite_environment()
    api_key = _required_secret("KITE_API_KEY")
    return KiteConnect(api_key=api_key).login_url()


def exchange_kite_request_token(value: str) -> str:
    """Exchange a request token or redirect URL and save the access token."""
    load_kite_environment()
    api_key = _required_secret("KITE_API_KEY")
    api_secret = _required_secret("KITE_API_SECRET")
    client = KiteConnect(api_key=api_key)
    request_token = _get_request_token(value)
    session = client.generate_session(request_token, api_secret)
    access_token = session["access_token"]

    success, key, _ = set_key(
        str(ENV_FILE),
        "KITE_ACCESS_TOKEN",
        access_token,
        quote_mode="always",
    )
    if not success or key != "KITE_ACCESS_TOKEN":
        raise OSError(f"Could not save the Kite access token to {ENV_FILE}.")
    os.environ["KITE_ACCESS_TOKEN"] = access_token
    return access_token


def login_to_kite() -> str:
    """Open Kite login, exchange the returned request token, and save the access token."""
    login_url = get_kite_login_url()
    print(f"Kite login URL: {login_url}")
    webbrowser.open(login_url)
    access_token = exchange_kite_request_token(
        getpass.getpass("Paste the redirected URL or request_token: ").strip()
    )
    print(f"Kite login successful. Access token saved to {ENV_FILE}.")
    return access_token
