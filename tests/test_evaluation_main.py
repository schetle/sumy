import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from sumy.evaluation.__main__ import app, handle_arguments, AVAILABLE_METHODS


class TestEvaluationMain(unittest.TestCase):
    def _make_ref_file(self, text="Reference sentence one. Reference sentence two."):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write(text)
        f.close()
        return f.name

    def test_handle_default_arguments(self):
        ref = self._make_ref_file()
        try:
            result = handle_arguments(
                method="lsa",
                reference_summary=ref,
                default_input_stream=StringIO("Whatever. This is a test sentence. Another sentence here."),
            )
            self.assertIsNotNone(result)
        finally:
            os.unlink(ref)

    def test_handle_wrong_format(self):
        ref = self._make_ref_file()
        try:
            self.assertRaises(
                ValueError,
                handle_arguments,
                method="lsa",
                reference_summary=ref,
                format="text",
                default_input_stream=StringIO("Whatever."),
            )
        finally:
            os.unlink(ref)

    def test_handle_url_and_file_raises(self):
        ref = self._make_ref_file()
        try:
            self.assertRaises(
                ValueError,
                handle_arguments,
                method="lsa",
                reference_summary=ref,
                url="http://example.com",
                file="/some/file.txt",
            )
        finally:
            os.unlink(ref)

    def test_handle_all_methods(self):
        ref = self._make_ref_file()
        try:
            for method in AVAILABLE_METHODS:
                result = handle_arguments(
                    method=method,
                    reference_summary=ref,
                    default_input_stream=StringIO(
                        "Whatever. This is a test sentence. Another sentence here. "
                        "Yet another sentence. One more sentence."
                    ),
                )
                self.assertIsNotNone(result)
        finally:
            os.unlink(ref)

    def test_handle_file_input_mode(self):
        content = b"This is a test sentence. Another sentence here."
        ref = self._make_ref_file()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            result = handle_arguments(
                method="lsa",
                reference_summary=ref,
                file=tmp_path,
            )
            self.assertIsNotNone(result)
        finally:
            os.unlink(ref)
            os.unlink(tmp_path)

    def test_handle_url_input_mode(self):
        ref = self._make_ref_file()
        try:
            mock_response = MagicMock()
            mock_response.read.return_value = b"<html><body><p>Test sentence here. Another one.</p></body></html>"
            mock_response.close.return_value = None
            with patch("sumy.evaluation.__main__.urllib.urlopen", return_value=mock_response):
                result = handle_arguments(
                    method="lsa",
                    reference_summary=ref,
                    url="http://example.com/article",
                )
            self.assertIsNotNone(result)
        finally:
            os.unlink(ref)

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("method", result.output.lower())

    def test_cli_no_args(self):
        runner = CliRunner()
        result = runner.invoke(app, [])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_invalid_method(self):
        runner = CliRunner()
        result = runner.invoke(app, ["invalidmethod", "/dev/null"])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_version(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0)
        from sumy import __version__
        self.assertIn(__version__, result.output)


if __name__ == "__main__":
    unittest.main()
