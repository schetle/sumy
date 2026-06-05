# -*- coding: utf-8 -*-
"""
Functional tests for PlaintextParser.

Verifies:
- Parses plain text strings correctly
- Handles bytes input (decodes to str)
- Produces correct document structure (paragraphs, sentences)
- Uses functools.cached_property for .document
- significant_words and stigma_words work
"""
import pytest
from sumy.parsers.plaintext import PlaintextParser
from sumy.models.dom import ObjectDocumentModel
from sumy.nlp.tokenizers import Tokenizer


TOKENIZER = Tokenizer("english")

SIMPLE_TEXT = (
    "Natural language processing is a field of computer science. "
    "It focuses on making computers understand human language.\n\n"
    "Machine learning is a subset of artificial intelligence. "
    "It enables computers to learn without being explicitly programmed."
)


class TestPlaintextParserFromString:
    """PlaintextParser.from_string — happy path."""

    def test_parse_basic_text(self):
        parser = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        doc = parser.document
        assert isinstance(doc, ObjectDocumentModel)

    def test_document_has_paragraphs(self):
        parser = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        assert len(parser.document.paragraphs) >= 1

    def test_document_has_sentences(self):
        parser = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        assert len(parser.document.sentences) >= 2

    def test_document_cached_property(self):
        """document property must be cached (functools.cached_property)."""
        parser = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        doc_first = parser.document
        doc_second = parser.document
        assert doc_first is doc_second

    def test_significant_words_cached(self):
        parser = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        sw_first = parser.significant_words
        sw_second = parser.significant_words
        assert sw_first is sw_second

    def test_stigma_words_cached(self):
        parser = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        stigma_first = parser.stigma_words
        stigma_second = parser.stigma_words
        assert stigma_first is stigma_second

    def test_significant_words_type(self):
        parser = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        assert isinstance(parser.significant_words, (tuple, frozenset))

    def test_stigma_words_type(self):
        parser = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        assert isinstance(parser.stigma_words, (tuple, frozenset))


class TestPlaintextParserBytesInput:
    """PlaintextParser handles bytes input (Python 3 bytes fix)."""

    def test_bytes_input_utf8(self):
        """Bytes input must be decoded to str without error."""
        text_bytes = SIMPLE_TEXT.encode("utf-8")
        parser = PlaintextParser(text_bytes, TOKENIZER)
        doc = parser.document
        assert isinstance(doc, ObjectDocumentModel)
        assert len(doc.sentences) >= 2

    def test_bytes_input_produces_same_result_as_str(self):
        """Bytes and str input must produce equivalent documents."""
        parser_str = PlaintextParser.from_string(SIMPLE_TEXT, TOKENIZER)
        parser_bytes = PlaintextParser(SIMPLE_TEXT.encode("utf-8"), TOKENIZER)
        sentences_str = [str(s) for s in parser_str.document.sentences]
        sentences_bytes = [str(s) for s in parser_bytes.document.sentences]
        assert sentences_str == sentences_bytes

    def test_bytes_with_non_ascii_content(self):
        """Bytes with UTF-8 encoded non-ASCII characters must decode correctly."""
        text = "Résumé is a French word. It means a summary of qualifications."
        text_bytes = text.encode("utf-8")
        parser = PlaintextParser(text_bytes, TOKENIZER)
        doc = parser.document
        assert len(doc.sentences) >= 1


class TestPlaintextParserHeadings:
    """PlaintextParser recognizes all-uppercase lines as headings."""

    def test_uppercase_line_becomes_heading(self):
        text = "INTRODUCTION\nThis is the first sentence. This is the second."
        parser = PlaintextParser.from_string(text, TOKENIZER)
        headings = parser.document.headings
        assert len(headings) == 1
        assert str(headings[0]) == "INTRODUCTION"

    def test_significant_words_from_heading(self):
        """If headings exist, significant_words returns words from headings."""
        text = "IMPORTANT CONCEPT\nThis discusses the important concept in detail."
        parser = PlaintextParser.from_string(text, TOKENIZER)
        sig_words = parser.significant_words
        # significant_words should contain words from the heading
        assert isinstance(sig_words, (tuple, frozenset))
        assert len(sig_words) > 0


class TestPlaintextParserFromFile:
    """PlaintextParser.from_file reads a file and parses it."""

    def test_from_file_readme(self, tmp_path):
        """from_file should open with UTF-8 encoding."""
        content = "First sentence here. Second sentence follows.\n\nThird paragraph sentence."
        test_file = tmp_path / "test.txt"
        test_file.write_text(content, encoding="utf-8")
        parser = PlaintextParser.from_file(str(test_file), TOKENIZER)
        doc = parser.document
        assert len(doc.sentences) >= 2

    def test_from_file_with_utf8_content(self, tmp_path):
        content = "Café au lait is delicious. Résumé writing requires effort."
        test_file = tmp_path / "utf8_test.txt"
        test_file.write_text(content, encoding="utf-8")
        parser = PlaintextParser.from_file(str(test_file), TOKENIZER)
        doc = parser.document
        assert len(doc.sentences) >= 1


class TestPlaintextParserBoundary:
    """Boundary conditions for PlaintextParser."""

    def test_single_sentence(self):
        parser = PlaintextParser.from_string("Just one sentence here.", TOKENIZER)
        doc = parser.document
        assert len(doc.sentences) == 1

    def test_multiple_paragraphs(self):
        text = "First para.\n\nSecond para.\n\nThird para."
        parser = PlaintextParser.from_string(text, TOKENIZER)
        # Each non-empty paragraph separated by blank lines
        assert len(parser.document.paragraphs) >= 2

    def test_empty_string_produces_valid_document(self):
        parser = PlaintextParser.from_string("", TOKENIZER)
        doc = parser.document
        assert isinstance(doc, ObjectDocumentModel)
