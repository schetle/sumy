import nltk.stem.snowball as nltk_stemmers_module

from .czech import stem_word as czech_stemmer


def null_stemmer(object):
    "Converts given object to unicode with lower letters."
    return str(object).lower()


class Stemmer:
    def __init__(self, language):
        self._stemmer = null_stemmer
        if language.lower() in ('czech', 'slovak'):
            self._stemmer = czech_stemmer
            return
        stemmer_classname = language.capitalize() + 'Stemmer'
        try:
            stemmer_class = getattr(nltk_stemmers_module, stemmer_classname)
        except AttributeError:
            raise LookupError(f"Stemmer is not available for language {language}.")
        self._stemmer = stemmer_class().stem

    def __call__(self, word):
        return self._stemmer(word)
