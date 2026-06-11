import sys

from functools import wraps
from os.path import dirname, abspath, join, exists


def cached_property(getter):
    """
    Decorator that converts a method into memoized property.
    The decorator works as expected only for classes with
    attribute '__dict__' and immutable properties.

    NOTE: We keep this custom implementation instead of using
    functools.cached_property because functools.cached_property
    does NOT work with __slots__-based classes (Sentence, Paragraph, etc.).
    """
    @wraps(getter)
    def decorator(self):
        key = "_cached_property_" + getter.__name__

        if not hasattr(self, key):
            setattr(self, key, getter(self))

        return getattr(self, key)

    return property(decorator)


def expand_resource_path(path):
    directory = dirname(sys.modules["sumy"].__file__)
    directory = abspath(directory)
    return join(directory, "data", str(path))


def get_stop_words(language):
    path = expand_resource_path(f"stopwords/{language}.txt")
    if not exists(path):
        raise LookupError(f"Stop-words are not available for language {language}.")
    return read_stop_words(path)


def read_stop_words(filename):
    with open(filename, "rb") as open_file:
        return frozenset(w.decode("utf-8").rstrip() for w in open_file.readlines())


class ItemsCount:
    def __init__(self, value):
        self._value = value

    def __call__(self, sequence):
        if isinstance(self._value, str):
            if self._value.endswith("%"):
                total_count = len(sequence)
                percentage = int(self._value[:-1])
                # at least one sentence should be choosen
                count = max(1, total_count*percentage // 100)
                return sequence[:count]
            else:
                return sequence[:int(self._value)]
        elif isinstance(self._value, (int, float)):
            return sequence[:int(self._value)]
        else:
            ValueError(f"Unsuported value of items count '{self._value}'.")

    def __repr__(self):
        return f"<ItemsCount: {self._value!r}>"
