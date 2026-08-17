# -*- coding: utf8 -*-

import sys

from os.path import dirname, abspath, join, exists


def expand_resource_path(path):
    directory = dirname(sys.modules["sumy"].__file__)
    directory = abspath(directory)
    return join(directory, str("data"), str(path))


def get_stop_words(language):
    path = expand_resource_path("stopwords/%s.txt" % language)
    if not exists(path):
        raise LookupError("Stop-words are not available for language %s." % language)
    return read_stop_words(path)


def read_stop_words(filename):
    with open(filename, "rb") as open_file:
        return frozenset(w.decode('utf-8').rstrip() for w in open_file.readlines())


def to_unicode(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


class ItemsCount(object):
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
            ValueError("Unsuported value of items count '%s'." % self._value)

    def __repr__(self):
        return str("<ItemsCount: %r>" % self._value)
