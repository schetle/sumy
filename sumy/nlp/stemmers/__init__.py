from __future__ import annotations

from collections.abc import Callable

import nltk.stem.snowball as nltk_stemmers_module

from .czech import stem_word as czech_stemmer


def null_stemmer(object: object) -> str:
    "Converts given object to unicode with lower letters."
    return (object.decode("utf-8") if isinstance(object, bytes) else str(object)).lower()


class Stemmer:
    def __init__(self, language: str) -> None:
        self._stemmer: Callable[[str], str] = null_stemmer
        if language.lower() in ('czech', 'slovak'):
            self._stemmer = czech_stemmer
            return
        stemmer_classname = language.capitalize() + 'Stemmer'
        try:
            stemmer_class = getattr(nltk_stemmers_module, stemmer_classname)
        except AttributeError:
            raise LookupError(f"Stemmer is not available for language {language}.")
        self._stemmer = stemmer_class().stem

    def __call__(self, word: str) -> str:
        return self._stemmer(word)
