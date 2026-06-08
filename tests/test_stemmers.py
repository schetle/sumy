import pytest

from sumy.nlp.stemmers import null_stemmer, Stemmer


def test_missing_stemmer_language():
    with pytest.raises(LookupError):
        Stemmer("klingon")


def test_null_stemmer():
    assert null_stemmer("ľŠčŤžÝáÍé") == "ľščťžýáíé"


def test_english_stemmer():
    english_stemmer = Stemmer('english')
    assert english_stemmer("beautiful") == "beauti"


def test_german_stemmer():
    german_stemmer = Stemmer('german')
    assert german_stemmer("sterben") == "sterb"


def test_czech_stemmer():
    czech_stemmer = Stemmer('czech')
    assert czech_stemmer("pěkný") == "pěkn"


def test_french_stemmer():
    french_stemmer = Stemmer('czech')
    assert french_stemmer("jolies") == "jol"


def test_slovak_stemmer():
    expected = Stemmer("czech")
    actual = Stemmer("slovak")
    assert type(actual) == type(expected)
    assert expected.__dict__ == actual.__dict__
