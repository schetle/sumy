import unittest
from io import StringIO

from typer.testing import CliRunner

from sumy.__main__ import app, handle_arguments, AVAILABLE_METHODS


class TestMain(unittest.TestCase):
    def test_handle_default_arguments(self):
        summarizer, parser, items_count = handle_arguments(
            method="lsa",
            default_input_stream=StringIO("Whatever. This is a test sentence. Another sentence here."),
        )
        self.assertIsNotNone(summarizer)
        self.assertIsNotNone(parser)

    def test_handle_wrong_format(self):
        self.assertRaises(
            ValueError,
            handle_arguments,
            method="lsa",
            url="http://example.com",
            format="text",  # Invalid format (should be "html" or "plaintext")
            default_input_stream=StringIO("Whatever."),
        )

    def test_handle_all_methods(self):
        for method in ["luhn", "lsa", "kl", "sum-basic"]:
            summarizer, parser, items_count = handle_arguments(
                method=method,
                default_input_stream=StringIO("Whatever. This is test text. Another sentence."),
            )
            self.assertIsNotNone(summarizer)

    def test_handle_valid_methods(self):
        for method in AVAILABLE_METHODS:
            summarizer, parser, items_count = handle_arguments(
                method=method,
                default_input_stream=StringIO(
                    "Whatever. This is a test sentence. Another sentence here. "
                    "Yet another sentence to summarize. One more sentence."
                ),
            )
            self.assertIsNotNone(summarizer)
            self.assertIsNotNone(parser)

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("method", result.output.lower())

    def test_cli_no_method(self):
        runner = CliRunner()
        result = runner.invoke(app, [])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_invalid_method(self):
        runner = CliRunner()
        result = runner.invoke(app, ["invalidmethod"])
        self.assertNotEqual(result.exit_code, 0)
