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

    def test_handle_file_input_mode(self):
        """handle_arguments reads from a file when --file is given."""
        import tempfile, os
        content = b"This is a test sentence. Another sentence here."
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            summarizer, parser, items_count = handle_arguments(
                method="lsa",
                file=tmp_path,
            )
            self.assertIsNotNone(summarizer)
            self.assertIsNotNone(parser)
        finally:
            os.unlink(tmp_path)

    def test_handle_url_and_file_raises(self):
        """Passing both --url and --file raises ValueError."""
        self.assertRaises(
            ValueError,
            handle_arguments,
            method="lsa",
            url="http://example.com",
            file="/some/file.txt",
        )

    def test_handle_url_input_mode(self):
        """handle_arguments uses urllib when --url is given."""
        from unittest.mock import patch, MagicMock
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html><body><p>This is a test sentence. Another one.</p></body></html>"
        mock_response.close.return_value = None
        with patch("sumy.__main__.urllib.urlopen", return_value=mock_response):
            summarizer, parser, items_count = handle_arguments(
                method="lsa",
                url="http://example.com/article",
                format="html",
            )
        self.assertIsNotNone(summarizer)

    def test_cli_version(self):
        """sumy --version prints the version string."""
        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0)
        from sumy import __version__
        self.assertIn(__version__, result.output)

    def test_cli_length_percentage(self):
        """--length=20% (percent sign in option value) is parsed correctly by typer."""
        runner = CliRunner()
        text = "First sentence here. Second sentence there. Third sentence present. Fourth sentence found. Fifth one too."
        result = runner.invoke(app, ["lsa", "--length=20%"], input=text)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(len(result.output.strip()) > 0)

    def test_cli_length_integer(self):
        """--length=2 (integer count) is parsed correctly."""
        runner = CliRunner()
        text = "First sentence here. Second sentence there. Third sentence present. Fourth sentence found. Fifth one too."
        result = runner.invoke(app, ["lsa", "--length=2"], input=text)
        self.assertEqual(result.exit_code, 0)
