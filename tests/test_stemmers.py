"""Tests for stemmers to verify they share the same API."""

import pytest

from sumy.nlp.stemmers import null_stemmer, Stemmer


def test_missing_stemmer_language():
    """Test that an unsupported language raises LookupError."""
    with pytest.raises(LookupError):
        Stemmer("klingon")


def test_null_stemmer():
    """Test that the null stemmer lowercases the input."""
    assert "ľščťžýáíé" == null_stemmer("ľŠčŤžÝáÍé")


def test_english_stemmer():
    """Test the English Snowball stemmer."""
    english_stemmer = Stemmer("english")
    assert "beauti" == english_stemmer("beautiful")


def test_german_stemmer():
    """Test the German Snowball stemmer."""
    german_stemmer = Stemmer("german")
    assert "sterb" == german_stemmer("sterben")


def test_czech_stemmer():
    """Test the Czech stemmer."""
    czech_stemmer = Stemmer("czech")
    assert "pěkn" == czech_stemmer("pěkný")


def test_french_stemmer():
    """Test the French stemmer (uses Czech stemmer)."""
    french_stemmer = Stemmer("czech")
    assert "jol" == french_stemmer("jolies")


def test_slovak_stemmer():
    """Test that Slovak stemmer is equivalent to Czech stemmer."""
    expected = Stemmer("czech")
    actual = Stemmer("slovak")
    assert type(actual) == type(expected)
    assert expected.__dict__ == actual.__dict__
