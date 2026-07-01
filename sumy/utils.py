from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from functools import wraps
from typing import Any


def cached_property(getter: Callable[..., Any]) -> property:
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


def expand_resource_path(path: str | Path) -> Path:
    directory = Path(sys.modules["sumy"].__file__).resolve().parent
    return directory / "data" / str(path)


def get_stop_words(language: str) -> frozenset[str]:
    path = expand_resource_path(f"stopwords/{language}.txt")
    if not path.exists():
        raise LookupError(f"Stop-words are not available for language {language}.")
    return read_stop_words(path)


def read_stop_words(filename: str | Path) -> frozenset[str]:
    with open(filename, "rb") as open_file:
        return frozenset(str(w.rstrip(), "utf-8") if isinstance(w, bytes) else w.rstrip() for w in open_file.readlines())


class ItemsCount:
    def __init__(self, value: int | float | str) -> None:
        self._value = value

    def __call__(self, sequence: Sequence[Any]) -> Sequence[Any]:
        if isinstance(self._value, (str, bytes)):
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
            raise ValueError(f"Unsupported value of items count '{self._value}'.")

    def __repr__(self) -> str:
        return f"<ItemsCount: {self._value!r}>"
