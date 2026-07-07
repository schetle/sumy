import nltk.stem.snowball as nltk_stemmers_module
from typing import Callable

from .czech import stem_word as czech_stemmer


def null_stemmer(word: str) -> str:
    "Converts given word to unicode with lower letters."
    return str(word).lower()


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
