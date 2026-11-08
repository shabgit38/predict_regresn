from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values

from stock_returns.auth import _get_request_token, login_to_kite


class KiteAuthTests(unittest.TestCase):
    def test_extracts_request_token_from_redirect_url_or_raw_token(self) -> None:
        self.assertEqual(
            _get_request_token("https://localhost/?status=success&request_token=abc123"),
            "abc123",
        )
        self.assertEqual(_get_request_token("abc123"), "abc123")

    def test_rejects_redirect_without_request_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not contain"):
            _get_request_token("https://localhost/?status=success")

    @patch.dict(os.environ, {}, clear=True)
    @patch("stock_returns.auth.KiteConnect")
    @patch("stock_returns.auth.webbrowser.open")
    @patch("stock_returns.auth.getpass.getpass", return_value="https://localhost/?request_token=req123")
    def test_login_saves_access_token_to_env_file(
        self,
        getpass_prompt: unittest.mock.Mock,
        webbrowser_open: unittest.mock.Mock,
        kite_connect: unittest.mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text("KITE_API_KEY=test-key\nKITE_API_SECRET=test-secret\n", encoding="utf-8")
            client = kite_connect.return_value
            client.login_url.return_value = "https://kite.example/login"
            client.generate_session.return_value = {"access_token": "new-access-token"}

            with patch("stock_returns.auth.ENV_FILE", env_file):
                access_token = login_to_kite()

            self.assertEqual(access_token, "new-access-token")
            getpass_prompt.assert_called_once()
            client.generate_session.assert_called_once_with("req123", "test-secret")
            webbrowser_open.assert_called_once_with("https://kite.example/login")
            self.assertEqual(dotenv_values(env_file)["KITE_ACCESS_TOKEN"], "new-access-token")
            self.assertEqual(os.environ["KITE_ACCESS_TOKEN"], "new-access-token")


if __name__ == "__main__":
    unittest.main()
