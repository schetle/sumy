"""Stemmer implementations for various languages."""

import nltk.stem.snowball as nltk_stemmers_module

from .czech import stem_word as czech_stemmer


def null_stemmer(obj: object) -> str:
    """Convert given object to lowercase unicode string.

    Args:
        obj: Object to convert.

    Returns:
        Lowercase string representation.
    """
    return str(obj).lower()


class Stemmer:
    """Language-aware stemmer dispatching to NLTK Snowball or Czech stemmer.

    Callable that stems a word according to the configured language.
    """

    def __init__(self, language: str):
        """Initialize a Stemmer for the given language.

        Args:
            language: Natural language name (e.g., 'english', 'czech').

        Raises:
            LookupError: If no stemmer is available for the language.
        """
        self._stemmer = null_stemmer
        if language.lower() in ("czech", "slovak"):
            self._stemmer = czech_stemmer
            return
        stemmer_classname = language.capitalize() + "Stemmer"
        try:
            stemmer_class = getattr(nltk_stemmers_module, stemmer_classname)
        except AttributeError:
            raise LookupError(f"Stemmer is not available for language {language}.")
        self._stemmer = stemmer_class().stem

    def __call__(self, word: str) -> str:
        """Stem a word.

        Args:
            word: Word to stem.

        Returns:
            Stemmed word.
        """
        return self._stemmer(word)
