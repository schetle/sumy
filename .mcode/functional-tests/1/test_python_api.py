# -*- coding: utf8 -*-
"""Functional tests for the public sumy library API, exercised directly in
Python (not via the CLI wrapper): PlaintextParser, Tokenizer, Stemmer, and
summarizers (LuhnSummarizer / LexRankSummarizer / KLSummarizer).

These import the real, installed `sumy` package and run its real pipeline
end-to-end against real text, matching the public usage pattern documented
in the project's README.
"""

import pytest

SAMPLE_TEXT = (
    "Artificial intelligence has transformed the way people interact with "
    "technology in daily life. Many industries, including healthcare, "
    "finance, and transportation, now rely heavily on machine learning "
    "models to automate decisions. These models are trained on large "
    "datasets that capture patterns humans might otherwise overlook. "
    "However, the rapid adoption of AI also raises important questions "
    "about fairness, privacy, and accountability."
)


def _build_parser_and_stemmer(language="english"):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.nlp.stemmers import Stemmer

    parser = PlaintextParser.from_string(SAMPLE_TEXT, Tokenizer(language))
    stemmer = Stemmer(language)
    return parser, stemmer


class TestPlaintextParserApi:
    def test_parses_real_text_into_sentences(self):
        parser, _ = _build_parser_and_stemmer()
        sentences = parser.document.sentences
        assert len(sentences) == 4
        assert str(sentences[0]).startswith("Artificial intelligence")

    def test_decodes_bytes_input_instead_of_stringifying(self):
        """Regression check for this milestone's fixed bug: passing bytes
        must decode as UTF-8 text, not produce a "b'...'" repr."""
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer

        parser = PlaintextParser(SAMPLE_TEXT.encode("utf-8"), Tokenizer("english"))
        text = str(parser.document.sentences[0])
        assert not text.startswith("b'")
        assert "Artificial intelligence" in text


class TestLuhnSummarizerApi:
    def test_luhn_summarizer_produces_real_summary(self):
        from sumy.summarizers.luhn import LuhnSummarizer
        from sumy.utils import get_stop_words

        parser, stemmer = _build_parser_and_stemmer()
        summarizer = LuhnSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")

        summary = summarizer(parser.document, 2)
        assert len(summary) == 2
        for sentence in summary:
            text = str(sentence)
            assert text.strip()
            assert " " in text


class TestLexRankSummarizerApi:
    def test_lex_rank_summarizer_produces_real_summary(self):
        from sumy.summarizers.lex_rank import LexRankSummarizer
        from sumy.utils import get_stop_words

        parser, stemmer = _build_parser_and_stemmer()
        summarizer = LexRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")

        summary = summarizer(parser.document, 2)
        assert len(summary) == 2
        for sentence in summary:
            assert str(sentence).strip()


class TestKLSummarizerApi:
    """Covers the same `kl` algorithm that crashes via the CLI, to confirm
    the failure is in the library's KLSummarizer itself and not something
    specific to the CLI wrapper's argument handling."""

    def test_kl_summarizer_on_realistic_multi_sentence_text(self):
        from sumy.summarizers.kl import KLSummarizer
        from sumy.utils import get_stop_words

        parser, stemmer = _build_parser_and_stemmer()
        summarizer = KLSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")

        # BUG: KLSummarizer._compute_ratings mixes unnormalized words from
        # `summary` (via _get_all_words_in_doc, which does not lowercase or
        # filter stop words) into the KL-divergence frequency comparison
        # against `word_freq` (which IS normalized/filtered). Any sentence
        # that contains a capitalized stop word (e.g. "At", "However") that
        # has already been picked into `summary` causes a KeyError in
        # `_kl_divergence`, because that exact-cased token was never added
        # to word_freq. This reproduces with ordinary, realistic English
        # text (not a contrived case) and is NOT specific to the --file
        # bug fixed earlier this milestone - it is a pre-existing defect
        # in sumy/summarizers/kl.py, also present on this milestone's
        # baseline ref, and is also exercised by the `sumy kl` CLI command.
        summary = summarizer(parser.document, 2)
        assert len(summary) == 2
