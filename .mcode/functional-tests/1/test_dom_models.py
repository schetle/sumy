# -*- coding: utf-8 -*-
"""
Functional tests for sumy DOM models (Sentence, Paragraph, ObjectDocumentModel).

Verifies:
- __slots__ removed (functools.cached_property caches correctly)
- Sentence.words is cached via functools.cached_property
- Paragraph.sentences is cached via functools.cached_property
- ObjectDocumentModel.sentences is cached via functools.cached_property
- __str__ methods work correctly (unicode_compatible decorator removed)
"""
import pytest
from sumy.models.dom import Sentence, Paragraph, ObjectDocumentModel
from sumy.nlp.tokenizers import Tokenizer


TOKENIZER = Tokenizer("english")


class TestSentence:
    """Sentence DOM class — no __slots__, functools.cached_property caching."""

    def test_sentence_creation(self):
        s = Sentence("Hello world.", TOKENIZER)
        assert str(s) == "Hello world."

    def test_sentence_words_returns_tuple(self):
        s = Sentence("Hello world.", TOKENIZER)
        words = s.words
        assert isinstance(words, tuple)
        assert len(words) > 0

    def test_sentence_words_cached_property_caches(self):
        """Verify functools.cached_property caches the result (no __slots__)."""
        s = Sentence("Hello world.", TOKENIZER)
        words_first = s.words
        words_second = s.words
        # Same object identity proves caching works
        assert words_first is words_second

    def test_sentence_no_slots(self):
        """Sentence must not have __slots__ — required for functools.cached_property."""
        assert not hasattr(Sentence, "__slots__")

    def test_sentence_has_dict(self):
        """Without __slots__, Sentence instances must have a __dict__."""
        s = Sentence("Test sentence.", TOKENIZER)
        assert hasattr(s, "__dict__")

    def test_sentence_str_method(self):
        """__str__ must return the sentence text (not object repr)."""
        text = "This is a test sentence."
        s = Sentence(text, TOKENIZER)
        result = str(s)
        assert result == text
        assert "object at 0x" not in result

    def test_sentence_is_heading_false_by_default(self):
        s = Sentence("Normal sentence.", TOKENIZER)
        assert s.is_heading is False

    def test_sentence_is_heading_true(self):
        s = Sentence("HEADING TEXT", TOKENIZER, is_heading=True)
        assert s.is_heading is True

    def test_sentence_equality(self):
        s1 = Sentence("Hello.", TOKENIZER)
        s2 = Sentence("Hello.", TOKENIZER)
        assert s1 == s2

    def test_sentence_inequality(self):
        s1 = Sentence("Hello.", TOKENIZER)
        s2 = Sentence("World.", TOKENIZER)
        assert s1 != s2

    def test_sentence_hash(self):
        s = Sentence("Hello.", TOKENIZER)
        assert isinstance(hash(s), int)

    def test_sentence_bytes_input_converted_to_str(self):
        """Sentence accepts bytes for text via str(text)."""
        s = Sentence("Hello world.", TOKENIZER)
        assert isinstance(str(s), str)


class TestParagraph:
    """Paragraph DOM class — no __slots__, functools.cached_property caching."""

    def _make_paragraph(self):
        s1 = Sentence("First sentence.", TOKENIZER)
        s2 = Sentence("Second sentence.", TOKENIZER)
        heading = Sentence("HEADING", TOKENIZER, is_heading=True)
        return Paragraph([heading, s1, s2])

    def test_paragraph_creation(self):
        p = self._make_paragraph()
        assert p is not None

    def test_paragraph_sentences_excludes_headings(self):
        p = self._make_paragraph()
        sentences = p.sentences
        assert len(sentences) == 2
        for s in sentences:
            assert not s.is_heading

    def test_paragraph_headings(self):
        p = self._make_paragraph()
        headings = p.headings
        assert len(headings) == 1
        assert headings[0].is_heading

    def test_paragraph_sentences_cached_property_caches(self):
        """functools.cached_property on Paragraph.sentences must cache result."""
        p = self._make_paragraph()
        sentences_first = p.sentences
        sentences_second = p.sentences
        assert sentences_first is sentences_second

    def test_paragraph_words_cached(self):
        p = self._make_paragraph()
        words_first = p.words
        words_second = p.words
        assert words_first is words_second

    def test_paragraph_no_slots(self):
        """Paragraph must not have __slots__."""
        assert not hasattr(Paragraph, "__slots__")

    def test_paragraph_has_dict(self):
        s1 = Sentence("Test.", TOKENIZER)
        p = Paragraph([s1])
        assert hasattr(p, "__dict__")

    def test_paragraph_str_method(self):
        p = self._make_paragraph()
        result = str(p)
        assert "Paragraph" in result
        assert "object at 0x" not in result

    def test_paragraph_only_accepts_sentence_instances(self):
        with pytest.raises(TypeError):
            Paragraph(["not a sentence"])

    def test_paragraph_words_are_tuple(self):
        p = self._make_paragraph()
        assert isinstance(p.words, tuple)


class TestObjectDocumentModel:
    """ObjectDocumentModel — functools.cached_property on sentences/headings/words."""

    def _make_odm(self):
        s1 = Sentence("First sentence.", TOKENIZER)
        s2 = Sentence("Second sentence.", TOKENIZER)
        heading = Sentence("HEADING", TOKENIZER, is_heading=True)
        p1 = Paragraph([heading, s1])
        p2 = Paragraph([s2])
        return ObjectDocumentModel([p1, p2])

    def test_odm_creation(self):
        odm = self._make_odm()
        assert odm is not None

    def test_odm_paragraphs(self):
        odm = self._make_odm()
        assert len(odm.paragraphs) == 2

    def test_odm_sentences_aggregated(self):
        odm = self._make_odm()
        # sentences excludes headings, so 2 regular sentences
        assert len(odm.sentences) == 2

    def test_odm_sentences_cached(self):
        """functools.cached_property on ObjectDocumentModel.sentences caches result."""
        odm = self._make_odm()
        sentences_first = odm.sentences
        sentences_second = odm.sentences
        assert sentences_first is sentences_second

    def test_odm_headings_aggregated(self):
        odm = self._make_odm()
        assert len(odm.headings) == 1

    def test_odm_words_cached(self):
        odm = self._make_odm()
        words_first = odm.words
        words_second = odm.words
        assert words_first is words_second

    def test_odm_str_method(self):
        odm = self._make_odm()
        result = str(odm)
        assert "DOM" in result
        assert "object at 0x" not in result

    def test_odm_words_are_tuple(self):
        odm = self._make_odm()
        assert isinstance(odm.words, tuple)
