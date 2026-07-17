from __future__ import annotations

import sys

from collections.abc import Sequence
from functools import cached_property
from os.path import dirname, abspath, join, exists


def _ensure_str(obj: object) -> str:
    return obj.decode("utf-8") if isinstance(obj, bytes) else str(obj)


def expand_resource_path(path: str) -> str:
    directory = dirname(sys.modules["sumy"].__file__)
    directory = abspath(directory)
    return join(directory, "data", str(path))


def get_stop_words(language: str) -> frozenset[str]:
    path = expand_resource_path("stopwords/%s.txt" % language)
    if not exists(path):
        raise LookupError("Stop-words are not available for language %s." % language)
    return read_stop_words(path)


def read_stop_words(filename: str) -> frozenset[str]:
    with open(filename, "rb") as open_file:
        return frozenset(_ensure_str(w.rstrip()) for w in open_file.readlines())


class ItemsCount(object):
    def __init__(self, value: str | int | float) -> None:
        self._value = value

    def __call__(self, sequence: Sequence) -> Sequence:
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
            ValueError("Unsuported value of items count '%s'." % self._value)

    def __repr__(self) -> str:
        return "<ItemsCount: %r>" % self._value
