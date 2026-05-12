import sys

from functools import wraps
from pathlib import Path


def cached_property(getter):
    """
    Decorator that converts a method into memoized property.
    The decorator works as expected only for classes with
    attribute '__dict__' and immutable properties.

    Note: This custom implementation is needed because the classes
    Sentence and Paragraph use __slots__ (no __dict__), so
    functools.cached_property cannot be used for them.
    """
    @wraps(getter)
    def decorator(self):
        key = "_cached_property_" + getter.__name__

        if not hasattr(self, key):
            setattr(self, key, getter(self))

        return getattr(self, key)

    return property(decorator)


def expand_resource_path(path):
    directory = Path(sys.modules["sumy"].__file__).parent.resolve()
    return str(directory / "data" / path)


def get_stop_words(language):
    path = expand_resource_path(f"stopwords/{language}.txt")
    if not Path(path).exists():
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
        return f"<ItemsCount: {self._value!r}>"
