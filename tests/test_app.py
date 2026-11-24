from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class BrowserAppTests(unittest.TestCase):
    def test_app_renders_input_form(self) -> None:
        app_path = Path(__file__).parent.parent / "stock_returns" / "app.py"
        app = AppTest.from_file(str(app_path)).run()

        self.assertFalse(app.exception)
        labels = {field.label for field in app.text_input}
        self.assertTrue({"Ticker", "Benchmark", "Exchange", "History duration"} <= labels)
        self.assertTrue(any(button.label == "Run evaluation" for button in app.button))


if __name__ == "__main__":
    unittest.main()
