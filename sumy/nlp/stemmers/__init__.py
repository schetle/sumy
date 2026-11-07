# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from collections.abc import Callable
from typing import Any

import nltk.stem.snowball as nltk_stemmers_module

from .czech import stem_word as czech_stemmer

from ..._compat import to_unicode


def null_stemmer(object: Any) -> str:
    "Converts given object to unicode with lower letters."
    return to_unicode(object).lower()


class Stemmer(object):
    def __init__(self, language: str) -> None:
        self._stemmer: Callable[[str], str] = null_stemmer
        if language.lower() in ('czech', 'slovak'):
            self._stemmer = czech_stemmer
            return
        stemmer_classname = language.capitalize() + 'Stemmer'
        try:
            stemmer_class = getattr(nltk_stemmers_module, stemmer_classname)
        except AttributeError:
            raise LookupError("Stemmer is not available for language %s." % language)
        self._stemmer = stemmer_class().stem

    def __call__(self, word: str) -> str:
        return self._stemmer(word)
