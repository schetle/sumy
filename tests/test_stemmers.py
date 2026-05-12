import pytest

from sumy.nlp.stemmers import null_stemmer, Stemmer


class TestStemmers:
    """Simple tests to make sure all stemmers share the same API."""

    def test_missing_stemmer_language(self):
        with pytest.raises(LookupError):
            Stemmer("klingon")

    def test_null_stemmer(self):
        assert "ľščťžýáíé" == null_stemmer("ľŠčŤžÝáÍé")

    def test_english_stemmer(self):
        english_stemmer = Stemmer('english')
        assert "beauti" == english_stemmer("beautiful")

    def test_german_stemmer(self):
        german_stemmer = Stemmer('german')
        assert "sterb" == german_stemmer("sterben")

    def test_czech_stemmer(self):
        czech_stemmer = Stemmer('czech')
        assert "pěkn" == czech_stemmer("pěkný")

    def test_french_stemmer(self):
        french_stemmer = Stemmer('czech')
        assert "jol" == french_stemmer("jolies")

    def test_slovak_stemmer(self):
        expected = Stemmer("czech")
        actual = Stemmer("slovak")
        assert type(actual) == type(expected)
        assert expected.__dict__ == actual.__dict__
