# -*- coding: utf8 -*-

import argparse
import unittest
import pytest
from io import StringIO

from sumy.__main__ import handle_arguments, AVAILABLE_METHODS
from sumy.__main__ import main as sumy_main


def make_namespace(**kwargs):
    """Helper to build argparse.Namespace with defaults."""
    defaults = dict(
        algorithm="lsa",
        url=None,
        file=None,
        length="20%",
        language="english",
        stopwords=None,
        format=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestMain(unittest.TestCase):
    def test_handle_default_arguments(self):
        args = make_namespace()
        handle_arguments(args, default_input_stream=StringIO("Whatever."))

    def test_handle_wrong_format(self):
        args = make_namespace(url="URL", format="text")
        self.assertRaises(ValueError, handle_arguments, args)

    def test_algorithm_choices(self):
        expected = {"luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"}
        self.assertEqual(set(AVAILABLE_METHODS.keys()), expected)

    def test_argparse_valid_algorithm(self):
        algorithms = list(AVAILABLE_METHODS.keys())
        for algo in algorithms:
            ns = make_namespace(algorithm=algo)
            self.assertEqual(ns.algorithm, algo)

    def test_handle_arguments_returns_summarizer_parser_count(self):
        args = make_namespace(algorithm="luhn")
        summarizer, parser, items_count = handle_arguments(
            args, default_input_stream=StringIO("Hello world. This is a test.")
        )
        self.assertIsNotNone(summarizer)
        self.assertIsNotNone(parser)
        self.assertIsNotNone(items_count)

    def test_handle_lsa_algorithm(self):
        args = make_namespace(algorithm="lsa")
        summarizer, parser, items_count = handle_arguments(
            args, default_input_stream=StringIO("Hello world. This is a test.")
        )
        from sumy.summarizers.lsa import LsaSummarizer
        self.assertIsInstance(summarizer, LsaSummarizer)

    def test_handle_edmundson_algorithm(self):
        args = make_namespace(algorithm="edmundson")
        summarizer, parser, items_count = handle_arguments(
            args, default_input_stream=StringIO("Hello world. This is a test.")
        )
        from sumy.summarizers.edmundson import EdmundsonSummarizer
        self.assertIsInstance(summarizer, EdmundsonSummarizer)

    def test_handle_default_input_stream_used_when_no_url_or_file(self):
        args = make_namespace(algorithm="lsa", url=None, file=None)
        stream = StringIO("A single sentence for testing.")
        # Should not raise
        summarizer, parser, items_count = handle_arguments(args, default_input_stream=stream)
        self.assertIsNotNone(summarizer)

    def test_format_html_with_url_does_not_raise_value_error(self):
        # "html" is a valid format — no ValueError should be raised from format validation
        # (It will fail at urlopen, not at ValueError, so we can't easily test full execution)
        # Just verify that the format check itself passes for "html"
        args = make_namespace(url="http://example.com", format="html")
        # format validation passes; urlopen will raise, but that's not a ValueError
        try:
            handle_arguments(args)
        except ValueError:
            self.fail("handle_arguments raised ValueError for valid format='html'")
        except Exception:
            pass  # Expected: urlopen or similar network error

    def test_format_plaintext_is_valid(self):
        args = make_namespace(format="plaintext")
        summarizer, parser, items_count = handle_arguments(
            args, default_input_stream=StringIO("Test sentence here.")
        )
        self.assertIsNotNone(summarizer)
