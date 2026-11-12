# -*- coding: utf8 -*-
"""Utility helpers shared across sumy.

Provides the ``cached_property`` decorator, helpers for locating bundled
resources and loading stop words, and the ``ItemsCount`` helper for
specifying how many items (absolute or percentage) a summary should contain.
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
    """
    Returns the absolute path to a resource bundled in the ``data``
    directory of the ``sumy`` package.

    :param path: Path of the resource relative to the ``data`` directory,
        e.g. ``"stopwords/english.txt"``.
    :returns: Absolute path to the resource. The existence of the file is
        not checked.
    :raises KeyError: When the ``sumy`` package has not been imported.
    """
    directory = dirname(sys.modules["sumy"].__file__)
    directory = abspath(directory)
    return join(directory, to_string("data"), to_string(path))


def get_stop_words(language):
    """
    Returns the stop-words bundled with sumy for the given language.

    :param language: Name of the language, e.g. ``"english"``.
    :returns: Immutable set of stop-words as unicode strings.
    :raises LookupError: When stop-words are not available for the language.
    """
    path = expand_resource_path("stopwords/%s.txt" % language)
    if not exists(path):
        raise LookupError("Stop-words are not available for language %s." % language)
    return read_stop_words(path)


def read_stop_words(filename):
    """
    Reads stop-words from a file with one stop-word per line.

    :param filename: Path to the file, which is read in binary mode.
    :returns: Immutable set of stop-words as unicode strings. Trailing
        whitespace of each line is stripped.
    :raises IOError: When the file can't be opened or read.
    """
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
