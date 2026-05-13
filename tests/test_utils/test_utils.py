"""Tests for utility functions: get_stop_words, read_stop_words, ItemsCount."""

import pytest

from sumy.utils import get_stop_words, read_stop_words, ItemsCount
from ..utils import expand_resource_path


def test_ok_stop_words_language():
    """Test that stop words load successfully for a valid language."""
    stop_words = get_stop_words("french")
    assert len(stop_words) > 1


def test_missing_stop_words_language():
    """Test that a missing language raises LookupError."""
    with pytest.raises(LookupError):
        get_stop_words("klingon")


def test_ok_custom_stopwords_file():
    """Test loading custom stop words from a file."""
    stop_words = read_stop_words(expand_resource_path("stopwords/language.txt"))
    assert len(stop_words) == 4


def test_custom_stop_words_file_not_found():
    """Test that a missing stop words file raises IOError."""
    with pytest.raises(IOError):
        read_stop_words(expand_resource_path("stopwords/klingon.txt"))


def test_percentage_items_count():
    """Test ItemsCount with percentage strings."""
    count = ItemsCount("20%")
    returned = count([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert returned == [0, 1]

    count = ItemsCount("100%")
    returned = count([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert returned == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    count = ItemsCount("50%")
    returned = count([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert returned == [0, 1, 2, 3, 4]

    count = ItemsCount("30%")
    returned = count([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert returned == [0, 1, 2]

    count = ItemsCount("35%")
    returned = count([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert returned == [0, 1, 2]


def test_float_items_count():
    """Test ItemsCount with float and boolean values."""
    count = ItemsCount(3.5)
    returned = count([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert returned == [0, 1, 2]

    count = ItemsCount(True)
    returned = count([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert returned == [0]

    count = ItemsCount(False)
    returned = count([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert returned == []


def test_unsupported_items_count():
    """Test ItemsCount with unsupported value raises ValueError."""
    count = ItemsCount("Hacker")
    with pytest.raises(ValueError):
        count([])
