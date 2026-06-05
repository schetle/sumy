# -*- coding: utf-8 -*-
"""
Functional tests for all 7 sumy summarizers.

Verifies:
- Each summarizer produces non-empty output on fixed input
- Requested sentence count is honored
- Luhn, LSA, TextRank, LexRank, SumBasic, Edmundson all work on a fixed 10-sentence document
- KL summarizer has a known pre-existing bug (documented test)
"""
import pytest
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import ItemsCount, get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer

# A fixed 10-sentence document used for all summarizer tests
DOCUMENT_TEXT = """\
Natural language processing is a subfield of computer science and artificial intelligence.
It enables computers to understand, interpret, and generate human language.
Machine learning algorithms are widely used in NLP tasks today.
Deep neural networks have greatly improved the accuracy of language models.
Text summarization reduces a long document to its most important points.
Automatic summarization has two main approaches: extractive and abstractive.
Extractive methods select important sentences directly from the original text.
Abstractive methods generate new sentences to represent the document content.
Modern summarization systems are evaluated using ROUGE scores and human judges.
Research in NLP continues to produce impressive results across many domains.
"""

LANGUAGE = "english"
TOKENIZER = Tokenizer(LANGUAGE)
STEMMER = Stemmer(LANGUAGE)
STOP_WORDS = get_stop_words(LANGUAGE)


def make_parser():
    return PlaintextParser.from_string(DOCUMENT_TEXT, TOKENIZER)


def make_standard_summarizer(cls):
    summarizer = cls(STEMMER)
    summarizer.stop_words = STOP_WORDS
    return summarizer


class TestLuhnSummarizer:
    """Luhn summarizer — most basic summarizer."""

    def test_produces_3_sentences(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LuhnSummarizer)
        result = summarizer(parser.document, 3)
        assert len(result) == 3

    def test_produces_1_sentence(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LuhnSummarizer)
        result = summarizer(parser.document, 1)
        assert len(result) == 1

    def test_sentences_are_from_document(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LuhnSummarizer)
        result = summarizer(parser.document, 3)
        document_sentences = set(str(s) for s in parser.document.sentences)
        for sentence in result:
            assert str(sentence) in document_sentences

    def test_sentences_are_non_empty(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LuhnSummarizer)
        result = summarizer(parser.document, 3)
        for sentence in result:
            assert len(str(sentence)) > 0

    def test_items_count_percentage(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LuhnSummarizer)
        count = ItemsCount("30%")
        result = summarizer(parser.document, count)
        assert len(result) >= 1


class TestLsaSummarizer:
    """LSA (Latent Semantic Analysis) summarizer."""

    def test_produces_3_sentences(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LsaSummarizer)
        result = summarizer(parser.document, 3)
        assert len(result) == 3

    def test_produces_1_sentence(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LsaSummarizer)
        result = summarizer(parser.document, 1)
        assert len(result) == 1

    def test_sentences_are_non_empty(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LsaSummarizer)
        result = summarizer(parser.document, 3)
        for sentence in result:
            assert len(str(sentence)) > 0

    def test_sentences_from_document(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LsaSummarizer)
        result = summarizer(parser.document, 3)
        document_sentences = set(str(s) for s in parser.document.sentences)
        for sentence in result:
            assert str(sentence) in document_sentences


class TestTextRankSummarizer:
    """TextRank summarizer."""

    def test_produces_3_sentences(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(TextRankSummarizer)
        result = summarizer(parser.document, 3)
        assert len(result) == 3

    def test_produces_1_sentence(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(TextRankSummarizer)
        result = summarizer(parser.document, 1)
        assert len(result) == 1

    def test_sentences_are_non_empty(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(TextRankSummarizer)
        result = summarizer(parser.document, 3)
        for sentence in result:
            assert len(str(sentence)) > 0

    def test_sentences_from_document(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(TextRankSummarizer)
        result = summarizer(parser.document, 3)
        document_sentences = set(str(s) for s in parser.document.sentences)
        for sentence in result:
            assert str(sentence) in document_sentences


class TestLexRankSummarizer:
    """LexRank summarizer."""

    def test_produces_3_sentences(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LexRankSummarizer)
        result = summarizer(parser.document, 3)
        assert len(result) == 3

    def test_produces_1_sentence(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LexRankSummarizer)
        result = summarizer(parser.document, 1)
        assert len(result) == 1

    def test_sentences_are_non_empty(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LexRankSummarizer)
        result = summarizer(parser.document, 3)
        for sentence in result:
            assert len(str(sentence)) > 0

    def test_sentences_from_document(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(LexRankSummarizer)
        result = summarizer(parser.document, 3)
        document_sentences = set(str(s) for s in parser.document.sentences)
        for sentence in result:
            assert str(sentence) in document_sentences


class TestSumBasicSummarizer:
    """SumBasic summarizer."""

    def test_produces_3_sentences(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(SumBasicSummarizer)
        result = summarizer(parser.document, 3)
        assert len(result) == 3

    def test_produces_1_sentence(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(SumBasicSummarizer)
        result = summarizer(parser.document, 1)
        assert len(result) == 1

    def test_sentences_are_non_empty(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(SumBasicSummarizer)
        result = summarizer(parser.document, 3)
        for sentence in result:
            assert len(str(sentence)) > 0

    def test_sentences_from_document(self):
        parser = make_parser()
        summarizer = make_standard_summarizer(SumBasicSummarizer)
        result = summarizer(parser.document, 3)
        document_sentences = set(str(s) for s in parser.document.sentences)
        for sentence in result:
            assert str(sentence) in document_sentences


class TestEdmundsonSummarizer:
    """Edmundson summarizer — requires null_words, bonus_words, stigma_words."""

    def test_produces_3_sentences(self):
        parser = make_parser()
        summarizer = EdmundsonSummarizer(STEMMER)
        summarizer.null_words = STOP_WORDS
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
        result = summarizer(parser.document, 3)
        assert len(result) == 3

    def test_produces_1_sentence(self):
        parser = make_parser()
        summarizer = EdmundsonSummarizer(STEMMER)
        summarizer.null_words = STOP_WORDS
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
        result = summarizer(parser.document, 1)
        assert len(result) == 1

    def test_sentences_are_non_empty(self):
        parser = make_parser()
        summarizer = EdmundsonSummarizer(STEMMER)
        summarizer.null_words = STOP_WORDS
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
        result = summarizer(parser.document, 3)
        for sentence in result:
            assert len(str(sentence)) > 0

    def test_sentences_from_document(self):
        parser = make_parser()
        summarizer = EdmundsonSummarizer(STEMMER)
        summarizer.null_words = STOP_WORDS
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
        result = summarizer(parser.document, 3)
        document_sentences = set(str(s) for s in parser.document.sentences)
        for sentence in result:
            assert str(sentence) in document_sentences


class TestKLSummarizer:
    """KL summarizer — has a known pre-existing KeyError bug."""

    def test_kl_raises_key_error_on_mixed_case_words(self):
        """
        Known pre-existing bug: _kl_divergence uses summary_freq keys
        (which may include raw/capitalized words from _get_all_words_in_doc)
        to look up in doc_freq (normalized/lowercased words from _compute_tf).
        This KeyError was present before the Python 3 modernization.
        """
        parser = make_parser()
        summarizer = make_standard_summarizer(KLSummarizer)
        with pytest.raises(KeyError):
            summarizer(parser.document, 3)

    def test_kl_summarizer_class_instantiates(self):
        """KLSummarizer class must still instantiate without error."""
        summarizer = KLSummarizer(STEMMER)
        summarizer.stop_words = STOP_WORDS
        assert summarizer is not None
