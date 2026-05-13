"""Language-dependent tokenizer wrapping NLTK punkt and word_tokenize."""

import re
import nltk


class Tokenizer:
    """Language-dependent tokenizer of text documents.

    Wraps NLTK's punkt sentence tokenizer and word tokenizer.
    """

    _WORD_PATTERN = re.compile(r"^[^\W\d_]+$", re.UNICODE)
    # feel free to contribute if you have better tokenizer for any of these languages :)
    LANGUAGE_ALIASES = {
        "slovak": "czech",
    }

    # improve tokenizer by adding specific abbreviations it has issues with
    # note the final point in these items must not be included
    LANGUAGE_EXTRA_ABREVS = {
        "english": ["e.g", "al", "i.e"],
        "german": ["al", "z.B", "Inc", "engl", "z. B", "vgl", "lat", "bzw", "S"],
    }

    def __init__(self, language: str):
        """Initialize a Tokenizer for the given language.

        Args:
            language: Natural language name (e.g., 'english', 'czech').
        """
        self._language = language

        tokenizer_language = self.LANGUAGE_ALIASES.get(language, language)
        self._sentence_tokenizer = self._get_sentence_tokenizer(tokenizer_language)

    @property
    def language(self) -> str:
        """Return the language of this tokenizer.

        Returns:
            Language name string.
        """
        return self._language

    def _get_sentence_tokenizer(self, language: str):
        """Load the NLTK punkt sentence tokenizer for the given language.

        Args:
            language: Language name for the tokenizer data.

        Returns:
            NLTK punkt tokenizer instance.
        """
        path = f"tokenizers/punkt_tab/{language}.pickle"
        return nltk.data.load(path)

    def to_sentences(self, paragraph: str) -> tuple:
        """Split a paragraph into sentences.

        Args:
            paragraph: Text to split into sentences.

        Returns:
            Tuple of sentence strings.
        """
        extra_abbreviations = self.LANGUAGE_EXTRA_ABREVS.get(self._language, [])
        self._sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
        sentences = self._sentence_tokenizer.tokenize(str(paragraph))
        return tuple(s.strip() for s in sentences)

    def to_words(self, sentence: str) -> tuple:
        """Split a sentence into words, filtering non-word tokens.

        Args:
            sentence: Text to split into words.

        Returns:
            Tuple of word strings.
        """
        words = nltk.word_tokenize(str(sentence))
        return tuple(filter(self._is_word, words))

    def _is_word(self, word: str) -> bool:
        """Check if a token is a real word (no digits or special characters).

        Args:
            word: Token to check.

        Returns:
            True if the token matches the word pattern.
        """
        return bool(Tokenizer._WORD_PATTERN.search(word))
