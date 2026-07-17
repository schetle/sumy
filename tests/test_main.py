import sys
import pytest
from io import StringIO
from argparse import Namespace

from sumy.__main__ import build_parser, handle_arguments, AVAILABLE_METHODS, PARSERS, __version__


def _make_args(**kwargs):
    """Build an argparse Namespace from keyword arguments with sensible defaults."""
    defaults = {
        'method': 'lsa',
        'length': '20%',
        'language': 'english',
        'stopwords': None,
        'format': None,
        'url': None,
        'file': None,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_parser_creates_successfully():
    parser = build_parser()
    assert parser is not None


def test_parser_valid_methods():
    parser = build_parser()
    for method in AVAILABLE_METHODS:
        args = parser.parse_args([method])
        assert args.method == method


def test_parser_default_length():
    parser = build_parser()
    args = parser.parse_args(['lsa'])
    assert args.length == '20%'


def test_parser_default_language():
    parser = build_parser()
    args = parser.parse_args(['lsa'])
    assert args.language == 'english'


def test_parser_invalid_method_raises():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['klingon_summarizer'])


def test_handle_default_arguments():
    args = _make_args()
    summarizer, doc_parser, items_count = handle_arguments(
        args, default_input_stream=StringIO("Whatever.")
    )
    assert summarizer is not None
    assert doc_parser is not None
    assert items_count is not None


def test_handle_wrong_format_raises():
    args = _make_args(url='http://example.com', format='text')
    with pytest.raises(ValueError):
        handle_arguments(args, default_input_stream=StringIO("Whatever."))


def test_handle_plaintext_format():
    args = _make_args(format='plaintext')
    summarizer, doc_parser, items_count = handle_arguments(
        args, default_input_stream=StringIO("This is a sentence.")
    )
    assert summarizer is not None


def test_version_accessible():
    from sumy import __version__ as pkg_version
    assert isinstance(pkg_version, str)
    assert len(pkg_version) > 0
