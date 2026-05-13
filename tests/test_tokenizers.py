"""Tests for the Tokenizer class."""

import pytest

from sumy.nlp.tokenizers import Tokenizer


def test_missing_language():
    """Test that an unsupported language raises LookupError."""
    with pytest.raises(LookupError):
        Tokenizer("klingon")


def test_ensure_czech_tokenizer_available():
    """Test that the Czech tokenizer is available and works correctly."""
    tokenizer = Tokenizer("czech")
    assert tokenizer.language == "czech"

    sentences = tokenizer.to_sentences("""
        Měl jsem sen, že toto je sen. Bylo to také zvláštní.
        Jakoby jsem plaval v moři rekurze.
    """)

    expected = (
        "Měl jsem sen, že toto je sen.",
        "Bylo to také zvláštní.",
        "Jakoby jsem plaval v moři rekurze.",
    )
    assert expected == sentences


def test_language_getter():
    """Test that the language getter returns the correct language."""
    tokenizer = Tokenizer("english")
    assert tokenizer.language == "english"


def test_tokenize_sentence():
    """Test word tokenization of a single sentence."""
    tokenizer = Tokenizer("english")
    words = tokenizer.to_words("I am a very nice sentence with comma, but..")

    expected = (
        "I", "am", "a", "very", "nice", "sentence",
        "with", "comma", "but",
    )
    assert expected == words


def test_tokenize_paragraph():
    """Test sentence tokenization of a paragraph."""
    tokenizer = Tokenizer("english")
    sentences = tokenizer.to_sentences("""
        I am a very nice sentence with comma, but..
        This is next sentence. "I'm bored", said Pepek.
        Ou jee, duffman is here.
    """)

    expected = (
        "I am a very nice sentence with comma, but..",
        "This is next sentence.",
        '"I\'m bored", said Pepek.',
        "Ou jee, duffman is here.",
    )
    assert expected == sentences


def test_slovak_alias_into_czech_tokenizer():
    """Test that Slovak uses the Czech tokenizer under the hood."""
    tokenizer = Tokenizer("slovak")
    assert tokenizer.language == "slovak"

    sentences = tokenizer.to_sentences("""
        Je to veľmi fajn. Bodaj by nie.
        Ale na druhej strane čo je to oproti inému?
        To nechám na čitateľa.
    """)

    expected = (
        "Je to veľmi fajn.",
        "Bodaj by nie.",
        "Ale na druhej strane čo je to oproti inému?",
        "To nechám na čitateľa.",
    )
    assert expected == sentences
