"""Utility functions and classes for the sumy package."""

import sys

from functools import cached_property
from os.path import dirname, abspath, join, exists


def expand_resource_path(path: str) -> str:
    """Expand a relative resource path to an absolute path within the sumy data directory.

    Args:
        path: Relative path within the data directory.

    Returns:
        Absolute path to the resource.
    """
    directory = dirname(sys.modules["sumy"].__file__)
    directory = abspath(directory)
    return join(directory, "data", path)


def get_stop_words(language: str) -> frozenset:
    """Load stop words for the given language.

    Args:
        language: Name of the language (e.g., 'english', 'czech').

    Returns:
        Frozenset of stop words.

    Raises:
        LookupError: If stop words are not available for the given language.
    """
    path = expand_resource_path(f"stopwords/{language}.txt")
    if not exists(path):
        raise LookupError(f"Stop-words are not available for language {language}.")
    return read_stop_words(path)


def read_stop_words(filename: str) -> frozenset:
    """Read stop words from a file.

    Args:
        filename: Path to the stop words file.

    Returns:
        Frozenset of stop words.
    """
    with open(filename, "rb") as open_file:
        return frozenset(w.rstrip().decode("utf8") for w in open_file.readlines())


class ItemsCount:
    """Callable that selects a number of items from a sequence.

    The count can be an absolute number or a percentage string (e.g., '20%').
    """

    def __init__(self, value):
        """Initialize ItemsCount with a count value.

        Args:
            value: Number of items or percentage string (e.g., '20%').
        """
        self._value = value

    def __call__(self, sequence):
        """Select items from the sequence based on the configured count.

        Args:
            sequence: The sequence to select items from.

        Returns:
            A slice of the sequence.

        Raises:
            ValueError: If the value type is unsupported.
        """
        if isinstance(self._value, str):
            if self._value.endswith("%"):
                total_count = len(sequence)
                percentage = int(self._value[:-1])
                # at least one sentence should be chosen
                count = max(1, total_count * percentage // 100)
                return sequence[:count]
            else:
                return sequence[:int(self._value)]
        elif isinstance(self._value, (int, float)):
            return sequence[:int(self._value)]
        else:
            raise ValueError(f"Unsupported value of items count '{self._value}'.")

    def __repr__(self):
        """Return string representation of ItemsCount."""
        return f"<ItemsCount: {self._value!r}>"
