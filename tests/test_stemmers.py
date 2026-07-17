import pytest

from sumy.nlp.stemmers import null_stemmer, Stemmer


def test_missing_stemmer_language():
    with pytest.raises(LookupError):
        Stemmer("klingon")


def test_null_stemmer():
    assert "ľščťžýáíé" == null_stemmer("ľŠčŤžÝáÍé")


def test_english_stemmer():
    english_stemmer = Stemmer('english')
    assert "beauti" == english_stemmer("beautiful")


def test_german_stemmer():
    german_stemmer = Stemmer('german')
    assert "sterb" == german_stemmer("sterben")


def test_czech_stemmer():
    czech_stemmer = Stemmer('czech')
    assert "pěkn" == czech_stemmer("pěkný")


def test_czech_stemmer_bytes_input():
    from sumy.nlp.stemmers.czech import stem_word
    assert stem_word("pěkný".encode("utf-8")) == stem_word("pěkný")


def test_french_stemmer():
    french_stemmer = Stemmer('czech')
    assert "jol" == french_stemmer("jolies")


def test_slovak_stemmer():
    expected = Stemmer("czech")
    actual = Stemmer("slovak")
    assert type(actual) == type(expected)
    assert expected.__dict__ == actual.__dict__
