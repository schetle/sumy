"""
Functional tests for PlaintextParser and HtmlParser programmatic APIs.
"""
from pathlib import Path
import pytest

_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "tests" / "data"


class TestPlaintextParser:
    """PlaintextParser.from_string() creates a document with sentences"""

    SAMPLE_TEXT = (
        "Natural language processing is a subfield of linguistics. "
        "It focuses on the interactions between computers and human language. "
        "The goal is to enable machines to understand text and speech. "
        "This field has grown rapidly with deep learning techniques."
    )

    def test_from_string_creates_document(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = PlaintextParser.from_string(self.SAMPLE_TEXT, Tokenizer("english"))
        assert parser.document is not None

    def test_document_has_sentences(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = PlaintextParser.from_string(self.SAMPLE_TEXT, Tokenizer("english"))
        assert len(parser.document.sentences) > 0

    def test_document_sentence_count(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = PlaintextParser.from_string(self.SAMPLE_TEXT, Tokenizer("english"))
        # Should have 4 sentences
        assert len(parser.document.sentences) == 4

    def test_sentence_has_words(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = PlaintextParser.from_string(self.SAMPLE_TEXT, Tokenizer("english"))
        first_sentence = parser.document.sentences[0]
        assert len(first_sentence.words) > 0

    def test_sentence_words_are_strings(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = PlaintextParser.from_string(self.SAMPLE_TEXT, Tokenizer("english"))
        for word in parser.document.sentences[0].words:
            assert isinstance(word, str)

    def test_empty_string_returns_empty_document(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = PlaintextParser.from_string("", Tokenizer("english"))
        assert len(parser.document.sentences) == 0

    def test_single_sentence_parsed(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = PlaintextParser.from_string(
            "This is a single sentence.", Tokenizer("english")
        )
        assert len(parser.document.sentences) == 1


class TestHtmlParser:
    """HtmlParser parses HTML bytes into a document with sentences"""

    HTML_FIXTURE = str(_DATA_ROOT / "snippets" / "paragraphs.html")

    @pytest.fixture(scope="module")
    def html_bytes(self):
        with open(self.HTML_FIXTURE, "rb") as f:
            return f.read()

    def test_html_parser_creates_document(self, html_bytes):
        from sumy.parsers.html import HtmlParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = HtmlParser(html_bytes, Tokenizer("czech"))
        assert parser.document is not None

    def test_html_document_has_sentences(self, html_bytes):
        from sumy.parsers.html import HtmlParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = HtmlParser(html_bytes, Tokenizer("czech"))
        assert len(parser.document.sentences) > 0

    def test_html_document_sentences_have_words(self, html_bytes):
        from sumy.parsers.html import HtmlParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = HtmlParser(html_bytes, Tokenizer("czech"))
        for sentence in parser.document.sentences:
            assert len(sentence.words) > 0

    def test_html_parser_significant_words(self, html_bytes):
        from sumy.parsers.html import HtmlParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = HtmlParser(html_bytes, Tokenizer("czech"))
        # significant_words should be a tuple/list/frozenset of strings
        sig_words = parser.significant_words
        assert sig_words is not None
        assert hasattr(sig_words, "__iter__")

    def test_html_parser_stigma_words(self, html_bytes):
        from sumy.parsers.html import HtmlParser
        from sumy.nlp.tokenizers import Tokenizer
        parser = HtmlParser(html_bytes, Tokenizer("czech"))
        # stigma_words should be iterable
        stig_words = parser.stigma_words
        assert stig_words is not None
        assert hasattr(stig_words, "__iter__")


class TestSummarizerProgrammatic:
    """Programmatic use of a summarizer with PlaintextParser"""

    SAMPLE_TEXT = (
        "Automatic text summarization is the process of reducing a text document. "
        "The goal is to produce a shorter version that preserves key information. "
        "Various algorithms exist for this task including extractive and abstractive methods. "
        "Extractive methods select existing sentences while abstractive methods generate new text."
    )

    def test_lsa_summarizer_returns_sentences(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        parser = PlaintextParser.from_string(self.SAMPLE_TEXT, Tokenizer("english"))
        summarizer = LsaSummarizer(Stemmer("english"))
        summarizer.stop_words = get_stop_words("english")
        sentences = summarizer(parser.document, 2)
        assert len(sentences) == 2

    def test_luhn_summarizer_returns_sentences(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.luhn import LuhnSummarizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        parser = PlaintextParser.from_string(self.SAMPLE_TEXT, Tokenizer("english"))
        summarizer = LuhnSummarizer(Stemmer("english"))
        summarizer.stop_words = get_stop_words("english")
        sentences = summarizer(parser.document, 1)
        assert len(sentences) == 1

    def test_summarizer_sentences_are_from_document(self):
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lsa import LsaSummarizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        parser = PlaintextParser.from_string(self.SAMPLE_TEXT, Tokenizer("english"))
        doc_sentences = set(str(s) for s in parser.document.sentences)
        summarizer = LsaSummarizer(Stemmer("english"))
        summarizer.stop_words = get_stop_words("english")
        sentences = summarizer(parser.document, 2)
        for s in sentences:
            assert str(s) in doc_sentences
