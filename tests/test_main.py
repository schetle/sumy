"""Tests for the sumy CLI argument parsing and handling."""

import pytest
from io import StringIO

from sumy.__main__ import build_parser, handle_arguments, main


def test_ok_args():
    """Test that valid arguments are parsed without error."""
    args = build_parser().parse_args(["luhn", "--url=http://example.com", "--format=html"])
    assert args.method == "luhn"
    assert args.url == "http://example.com"
    assert args.format == "html"


def test_args_no_method():
    """Test that missing method argument causes SystemExit."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_args_just_method():
    """Test that a method alone uses default values."""
    args = build_parser().parse_args(["lsa"])
    assert args.method == "lsa"
    assert args.url is None
    assert args.file is None
    assert args.format is None
    assert args.language == "english"
    assert args.length == "20%"
    assert args.stopwords is None


def test_args_invalid_method():
    """Test that an invalid method causes SystemExit."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["invalid_method"])


def test_handle_default_arguments():
    """Test that handle_arguments works with default args and stdin input."""
    args = build_parser().parse_args(["lsa"])
    summarizer, parser, items_count = handle_arguments(
        args, default_input_stream=StringIO("Whatever. This is a test.")
    )
    assert summarizer is not None
    assert parser is not None
    assert items_count is not None


def test_all_methods_accepted():
    """Test that all supported methods are accepted by the parser."""
    methods = ["luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"]
    for method in methods:
        args = build_parser().parse_args([method])
        assert args.method == method


def test_args_with_language():
    """Test that the --language option is parsed correctly."""
    args = build_parser().parse_args(["luhn", "--language", "czech"])
    assert args.language == "czech"


def test_args_with_length():
    """Test that the --length option is parsed correctly."""
    args = build_parser().parse_args(["luhn", "--length", "5"])
    assert args.length == "5"


def test_args_with_file():
    """Test that the --file option is parsed correctly."""
    args = build_parser().parse_args(["luhn", "--file", "/path/to/file.txt"])
    assert args.file == "/path/to/file.txt"
