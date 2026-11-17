import sys
from collections.abc import Callable, Sequence
from functools import wraps
from os.path import abspath, dirname, exists, join
from typing import Any


def cached_property(getter: Callable[[Any], Any]) -> property:
    """
    Decorator that converts a method into memoized property.
    The decorator works as expected only for classes with
    attribute '__dict__' and immutable properties.
    """
    @wraps(getter)
    def decorator(self: Any) -> Any:
        key = "_cached_property_" + getter.__name__

        if not hasattr(self, key):
            setattr(self, key, getter(self))

        return getattr(self, key)

    return property(decorator)


def expand_resource_path(path: str) -> str:
    directory = dirname(sys.modules["sumy"].__file__)
    directory = abspath(directory)
    return join(directory, "data", path)


def get_stop_words(language: str) -> frozenset[str]:
    path = expand_resource_path(f"stopwords/{language}.txt")
    if not exists(path):
        raise LookupError(f"Stop-words are not available for language {language}.")
    return read_stop_words(path)


def decode_utf8(value: str | bytes) -> str:
    return value.decode("utf8") if isinstance(value, bytes) else value


def read_stop_words(filename: str) -> frozenset[str]:
    with open(filename, "rb") as open_file:
        return frozenset(decode_utf8(w.rstrip()) for w in open_file.readlines())


class ItemsCount:
    def __init__(self, value: str | int | float) -> None:
        self._value = value

    def __call__(self, sequence: Sequence[Any]) -> Sequence[Any] | None:
        if isinstance(self._value, str):
            if self._value.endswith("%"):
                total_count = len(sequence)
                percentage = int(self._value[:-1])
                # at least one sentence should be choosen
                count = max(1, total_count * percentage // 100)
                return sequence[:count]
            else:
                return sequence[:int(self._value)]
        elif isinstance(self._value, (int, float)):
            return sequence[:int(self._value)]
        else:
            ValueError(f"Unsuported value of items count '{self._value}'.")

    def __repr__(self) -> str:
        return f"<ItemsCount: {self._value!r}>"
