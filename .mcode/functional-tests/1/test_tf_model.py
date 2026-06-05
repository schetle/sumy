# -*- coding: utf-8 -*-
"""
Functional tests for TfDocumentModel.

Verifies:
- collections.abc.Sequence is used (not removed collections.Sequence)
- TfDocumentModel works correctly with list of words
- TfDocumentModel works with string + tokenizer
- term frequency, normalized frequency, magnitude work correctly
"""
import pytest
from collections.abc import Sequence
from sumy.models.tf import TfDocumentModel
from sumy.nlp.tokenizers import Tokenizer


TOKENIZER = Tokenizer("english")


class TestTfDocumentModelHappyPath:
    """TfDocumentModel — basic happy path usage."""

    def test_create_from_word_list(self):
        words = ["hello", "world", "hello", "python"]
        model = TfDocumentModel(words)
        assert model is not None

    def test_terms_property(self):
        words = ["hello", "world", "hello"]
        model = TfDocumentModel(words)
        terms = list(model.terms)
        assert "hello" in terms
        assert "world" in terms

    def test_term_frequency(self):
        words = ["hello", "hello", "world"]
        model = TfDocumentModel(words)
        assert model.term_frequency("hello") == 2
        assert model.term_frequency("world") == 1
        assert model.term_frequency("notpresent") == 0

    def test_term_frequency_case_insensitive(self):
        """TfDocumentModel lowercases all words."""
        words = ["Hello", "WORLD", "hello"]
        model = TfDocumentModel(words)
        assert model.term_frequency("hello") == 2
        assert model.term_frequency("world") == 1

    def test_normalized_term_frequency(self):
        words = ["hello", "hello", "world"]
        model = TfDocumentModel(words)
        # hello appears 2 times, world 1 time; max freq is 2
        # normalized = smooth + (1-smooth)*freq/max = 0.0 + 1.0*2/2 = 1.0
        ntf = model.normalized_term_frequency("hello")
        assert abs(ntf - 1.0) < 1e-9

    def test_normalized_term_frequency_with_smooth(self):
        words = ["hello", "hello", "world"]
        model = TfDocumentModel(words)
        # world freq=1, max=2, with smooth=0.4: 0.4 + 0.6*1/2 = 0.7
        ntf = model.normalized_term_frequency("world", smooth=0.4)
        assert abs(ntf - 0.7) < 1e-9

    def test_magnitude_property(self):
        words = ["a", "b", "c"]
        model = TfDocumentModel(words)
        assert model.magnitude > 0

    def test_most_frequent_terms(self):
        words = ["hello", "hello", "hello", "world", "world", "python"]
        model = TfDocumentModel(words)
        top2 = model.most_frequent_terms(2)
        assert top2[0] == "hello"
        assert top2[1] == "world"

    def test_most_frequent_terms_count_zero_returns_all(self):
        words = ["a", "b", "c"]
        model = TfDocumentModel(words)
        all_terms = model.most_frequent_terms(0)
        assert len(all_terms) == 3

    def test_create_from_string_with_tokenizer(self):
        model = TfDocumentModel("Hello world hello python", TOKENIZER)
        assert model.term_frequency("hello") == 2


class TestTfDocumentModelBoundary:
    """TfDocumentModel — edge cases and boundary conditions."""

    def test_empty_word_list(self):
        """Empty list should create a model with no terms and max_frequency=1."""
        model = TfDocumentModel([])
        assert list(model.terms) == []
        # magnitude of empty doc
        import math
        assert model.magnitude == 0.0

    def test_single_word(self):
        model = TfDocumentModel(["hello"])
        assert model.term_frequency("hello") == 1
        assert abs(model.normalized_term_frequency("hello") - 1.0) < 1e-9

    def test_many_repeated_words(self):
        words = ["test"] * 100
        model = TfDocumentModel(words)
        assert model.term_frequency("test") == 100

    def test_negative_count_raises(self):
        model = TfDocumentModel(["a", "b"])
        with pytest.raises(ValueError):
            model.most_frequent_terms(-1)


class TestTfDocumentModelInvalidInput:
    """TfDocumentModel — invalid input handling."""

    def test_string_without_tokenizer_raises(self):
        with pytest.raises(ValueError):
            TfDocumentModel("hello world")

    def test_non_sequence_raises(self):
        with pytest.raises(ValueError):
            TfDocumentModel(42)

    def test_non_sequence_dict_raises(self):
        with pytest.raises(ValueError):
            TfDocumentModel({"word": 1})


class TestCollectionsAbcImport:
    """Verify collections.abc.Sequence is used (Python 3.10+ compatibility fix)."""

    def test_list_is_abc_sequence(self):
        """TfDocumentModel accepts list because list is a Sequence."""
        words = ["hello", "world"]
        model = TfDocumentModel(words)
        assert model is not None

    def test_tuple_is_abc_sequence(self):
        """TfDocumentModel accepts tuple because tuple is a Sequence."""
        words = ("hello", "world")
        model = TfDocumentModel(words)
        assert model is not None

    def test_collections_abc_import_works(self):
        """Verify collections.abc.Sequence import works (not deprecated collections.Sequence)."""
        # This import must not raise an ImportError on Python 3.10+
        from collections.abc import Sequence as AbcSequence
        assert issubclass(list, AbcSequence)
