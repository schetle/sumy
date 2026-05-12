import pytest

from sumy.nlp.tokenizers import Tokenizer
from sumy.models import TfDocumentModel


class TestTfModel:
    def test_no_tokenizer_with_string(self):
        with pytest.raises(ValueError):
            TfDocumentModel("text without tokenizer")

    def test_pretokenized_words(self):
        model = TfDocumentModel(("wA", "WB", "wB", "WA"))

        terms = tuple(sorted(model.terms))
        assert terms == ("wa", "wb")

    def test_pretokenized_words_frequencies(self):
        model = TfDocumentModel(("wC", "wC", "WC", "wA", "WB", "wB"))

        assert model.term_frequency("wa") == 1
        assert model.term_frequency("wb") == 2
        assert model.term_frequency("wc") == 3
        assert model.term_frequency("wd") == 0

        assert model.most_frequent_terms() == ("wc", "wb", "wa")

    def test_magnitude(self):
        tokenizer = Tokenizer("english")
        text = "wA wB wC wD"
        model = TfDocumentModel(text, tokenizer)

        assert model.magnitude == pytest.approx(2.0)

    def test_terms(self):
        tokenizer = Tokenizer("english")
        text = "wA wB wC wD wB wD wE"
        model = TfDocumentModel(text, tokenizer)

        terms = tuple(sorted(model.terms))
        assert terms == ("wa", "wb", "wc", "wd", "we")

    def test_term_frequency(self):
        tokenizer = Tokenizer("english")
        text = "wA wB wC wA wA wC wD wCwB"
        model = TfDocumentModel(text, tokenizer)

        assert model.term_frequency("wa") == 3
        assert model.term_frequency("wb") == 1
        assert model.term_frequency("wc") == 2
        assert model.term_frequency("wd") == 1
        assert model.term_frequency("wcwb") == 1
        assert model.term_frequency("we") == 0
        assert model.term_frequency("missing") == 0

    def test_most_frequent_terms(self):
        tokenizer = Tokenizer("english")
        text = "wE wD wC wB wA wE WD wC wB wE wD WE wC wD wE"
        model = TfDocumentModel(text, tokenizer)

        assert model.most_frequent_terms(1) == ("we",)
        assert model.most_frequent_terms(2) == ("we", "wd")
        assert model.most_frequent_terms(3) == ("we", "wd", "wc")
        assert model.most_frequent_terms(4) == ("we", "wd", "wc", "wb")
        assert model.most_frequent_terms(5) == ("we", "wd", "wc", "wb", "wa")
        assert model.most_frequent_terms() == ("we", "wd", "wc", "wb", "wa")

    def test_most_frequent_terms_empty(self):
        tokenizer = Tokenizer("english")
        model = TfDocumentModel("", tokenizer)

        assert model.most_frequent_terms() == ()
        assert model.most_frequent_terms(10) == ()

    def test_most_frequent_terms_negative_count(self):
        tokenizer = Tokenizer("english")
        model = TfDocumentModel("text", tokenizer)

        with pytest.raises(ValueError):
            model.most_frequent_terms(-1)

    def test_normalized_words_frequencies(self):
        words = "a b c d e c b d c e e d e d e".split()
        model = TfDocumentModel(tuple(words))

        assert model.normalized_term_frequency("a") == pytest.approx(1/5)
        assert model.normalized_term_frequency("b") == pytest.approx(2/5)
        assert model.normalized_term_frequency("c") == pytest.approx(3/5)
        assert model.normalized_term_frequency("d") == pytest.approx(4/5)
        assert model.normalized_term_frequency("e") == pytest.approx(5/5)
        assert model.normalized_term_frequency("z") == pytest.approx(0.0)

        assert model.most_frequent_terms() == ("e", "d", "c", "b", "a")

    def test_normalized_words_frequencies_with_smoothing_term(self):
        words = "a b c d e c b d c e e d e d e".split()
        model = TfDocumentModel(tuple(words))

        assert model.normalized_term_frequency("a", 0.5) == pytest.approx(0.5 + 1/10)
        assert model.normalized_term_frequency("b", 0.5) == pytest.approx(0.5 + 2/10)
        assert model.normalized_term_frequency("c", 0.5) == pytest.approx(0.5 + 3/10)
        assert model.normalized_term_frequency("d", 0.5) == pytest.approx(0.5 + 4/10)
        assert model.normalized_term_frequency("e", 0.5) == pytest.approx(0.5 + 5/10)
        assert model.normalized_term_frequency("z", 0.5) == pytest.approx(0.5)

        assert model.most_frequent_terms() == ("e", "d", "c", "b", "a")
