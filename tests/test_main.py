import argparse
import unittest
from io import StringIO

import pytest

from sumy.__main__ import handle_arguments, main


class TestMain(unittest.TestCase):
    def test_main_with_valid_method(self):
        """Test that main() runs without error for a valid method with file input."""
        # Use a test file that exists
        exit_code = main(["luhn", "--file", "tests/data/snippets/prevko.txt", "--language", "czech", "--length", "3"])
        assert exit_code == 0

    def test_main_no_args_exits(self):
        """Test that calling main with no args raises SystemExit (argparse requires method)."""
        with pytest.raises(SystemExit):
            main([])

    def test_main_invalid_method_exits(self):
        """Test that an invalid method name raises SystemExit."""
        with pytest.raises(SystemExit):
            main(["invalid_method"])

    def test_main_version(self):
        """Test that --version raises SystemExit (argparse behavior)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_handle_arguments_default_input(self):
        """Test handle_arguments with default input (stdin)."""
        args = argparse.Namespace(
            method="lsa",
            url=None,
            file=None,
            format=None,
            length="20%",
            language="english",
            stopwords=None,
        )
        summarizer, parser, items_count = handle_arguments(args, default_input_stream=StringIO("This is a test sentence. And another one."))
        assert summarizer is not None
        assert parser is not None

    def test_handle_arguments_wrong_format(self):
        """Test that handle_arguments raises ValueError for invalid format."""
        args = argparse.Namespace(
            method="lsa",
            url="http://example.com",
            file=None,
            format="text",
            length="20%",
            language="english",
            stopwords=None,
        )
        with pytest.raises(ValueError):
            handle_arguments(args, default_input_stream=StringIO("Whatever."))

    def test_handle_arguments_with_file(self):
        """Test handle_arguments with --file argument."""
        args = argparse.Namespace(
            method="luhn",
            url=None,
            file="tests/data/snippets/prevko.txt",
            format=None,
            length="20%",
            language="czech",
            stopwords=None,
        )
        summarizer, parser, items_count = handle_arguments(args)
        assert summarizer is not None
        assert parser is not None
