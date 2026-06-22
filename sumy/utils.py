from __future__ import annotations

import sys

from functools import cached_property  # noqa: F401
from os.path import dirname, abspath, join, exists
from typing import Any


def expand_resource_path(path: str) -> str:
    module_file = sys.modules["sumy"].__file__
    assert module_file is not None
    directory = dirname(module_file)
    directory = abspath(directory)
    return join(directory, str("data"), str(path))


def get_stop_words(language: str) -> frozenset[str]:
    path = expand_resource_path("stopwords/%s.txt" % language)
    if not exists(path):
        raise LookupError("Stop-words are not available for language %s." % language)
    return read_stop_words(path)


def read_stop_words(filename: str) -> frozenset[str]:
    with open(filename, "rb") as open_file:
        return frozenset(w.rstrip().decode("utf8") for w in open_file.readlines())


class ItemsCount(object):
    def __init__(self, value: str | int | float) -> None:
        self._value = value

    def __call__(self, sequence: Any) -> Any:
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
            raise ValueError("Unsuported value of items count '%s'." % self._value)

    def __repr__(self) -> str:
        return str("<ItemsCount: %r>" % self._value)
