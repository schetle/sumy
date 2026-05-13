"""Tests for the LSA summarizer."""

import pytest

from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from ..utils import build_document, load_resource


def test_dictionary_without_stop_words():
    """Test that stop words are excluded from the LSA dictionary."""
    summarizer = LsaSummarizer()
    summarizer.stop_words = ["stop", "Halt", "SHUT", "HmMm"]

    document = build_document(
        ("stop halt shut hmmm", "Stop Halt Shut Hmmm",),
        ("StOp HaLt ShUt HmMm", "STOP HALT SHUT HMMM",),
        ("Some relevant sentence", "Some moRe releVant sentEnce",),
    )

    expected = frozenset(["some", "more", "relevant", "sentence"])
    dictionary = summarizer._create_dictionary(document)
    assert expected == frozenset(dictionary.keys())


def test_empty_document():
    """Test that an empty document returns no sentences."""
    document = build_document()
    summarizer = LsaSummarizer()

    sentences = summarizer(document, 10)
    assert len(sentences) == 0


def test_single_sentence():
    """Test LSA summarization with a single sentence."""
    document = build_document(("I am the sentence you like",))
    summarizer = LsaSummarizer()
    summarizer.stopwords = ("I", "am", "the",)

    sentences = summarizer(document, 10)
    assert len(sentences) == 1
    assert str(sentences[0]) == "I am the sentence you like"


def test_document():
    """Test LSA summarization with a multi-paragraph document."""
    document = build_document(
        ("I am the sentence you like", "Do you like me too",),
        ("This sentence is better than that above", "Are you kidding me",)
    )
    summarizer = LsaSummarizer()
    summarizer.stopwords = (
        "I", "am", "the", "you", "are", "me", "is", "than", "that", "this",
    )

    sentences = summarizer(document, 2)
    assert len(sentences) == 2
    assert str(sentences[0]) == "I am the sentence you like"
    assert str(sentences[1]) == "This sentence is better than that above"


def test_real_example():
    """Test LSA summarizer on a real Czech text.

    Source: http://www.prevko.cz/dite/skutecne-pribehy-deti
    """
    parser = PlaintextParser.from_string(
        load_resource("snippets/prevko.txt"),
        Tokenizer("czech")
    )
    summarizer = LsaSummarizer(Stemmer("czech"))
    summarizer.stop_words = get_stop_words("czech")

    sentences = summarizer(parser.document, 2)
    assert len(sentences) == 2


def test_article_example():
    """Test LSA summarizer on a longer Czech article.

    Source: http://www.prevko.cz/dite/skutecne-pribehy-deti
    """
    parser = PlaintextParser.from_string(
        load_resource("articles/prevko_cz_1.txt"),
        Tokenizer("czech")
    )
    summarizer = LsaSummarizer(Stemmer("czech"))
    summarizer.stop_words = get_stop_words("czech")

    sentences = summarizer(parser.document, 20)
    assert len(sentences) == 20


def test_issue_5_svd_converges():
    """Test for SVD convergence issue.

    Source: https://github.com/miso-belica/sumy/issues/5
    """
    pytest.skip("Can't reproduce the issue.")

    parser = PlaintextParser.from_string(
        load_resource("articles/svd_converges.txt"),
        Tokenizer("english")
    )
    summarizer = LsaSummarizer(Stemmer("english"))
    summarizer.stop_words = get_stop_words("english")

    sentences = summarizer(parser.document, 20)
    assert len(sentences) == 20
