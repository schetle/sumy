# -*- coding: utf8 -*-
"""
Utility helpers shared across the sumy package.

Public helpers:

- cached_property: decorator turning a method into a memoized property.
- expand_resource_path: resolves a path inside the bundled "data" directory.
- get_stop_words: loads the stop-words of a language from bundled resources.
- read_stop_words: reads a stop-words file into a frozenset of strings.
- ItemsCount: callable that trims a sequence to a count or a percentage.
"""

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import sys

from functools import wraps
from os.path import dirname, abspath, join, exists
from ._compat import to_string, to_unicode, string_types


def cached_property(getter):
    """
    Decorator that converts a method into memoized property.
    The decorator works as expected only for classes with
    attribute '__dict__' and immutable properties.
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
    return join(directory, to_string("data"), to_string(path))


def get_stop_words(language):
    path = expand_resource_path("stopwords/%s.txt" % language)
    if not exists(path):
        raise LookupError("Stop-words are not available for language %s." % language)
    return read_stop_words(path)


def read_stop_words(filename):
    with open(filename, "rb") as open_file:
        return frozenset(to_unicode(w.rstrip()) for w in open_file.readlines())


class ItemsCount(object):
    def __init__(self, value):
        self._value = value

    def __call__(self, sequence):
        if isinstance(self._value, string_types):
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

    def __repr__(self):
        return to_string("<ItemsCount: %r>" % self._value)
