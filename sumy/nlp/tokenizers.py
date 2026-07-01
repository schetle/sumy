
import re
from typing import Any

import nltk

class Tokenizer:
    """Language dependent tokenizer of text document."""

    _WORD_PATTERN: re.Pattern[str] = re.compile(r"^[^\W\d_]+$", re.UNICODE)
    # feel free to contribute if you have better tokenizer for any of these languages :)
    LANGUAGE_ALIASES: dict[str, str] = {
        "slovak": "czech",
    }

    # improve tokenizer by adding specific abbreviations it has issues with
    # note the final point in these items must not be included
    LANGUAGE_EXTRA_ABREVS: dict[str, list[str]] = {
        "english": ['e.g', 'al', 'i.e'],
        "german": ['al', 'z.B', 'Inc','engl','z. B', 'vgl', 'lat', 'bzw', 'S'],
    }

    def __init__(self, language: str) -> None:
        self._language: str = language

        tokenizer_language = self.LANGUAGE_ALIASES.get(language, language)
        self._sentence_tokenizer: Any = self._sentence_tokenizer(tokenizer_language)

    @property
    def language(self) -> str:
        return self._language

    def _sentence_tokenizer(self, language: str) -> Any:
        path = f"tokenizers/punkt_tab/{language}.pickle"
        return nltk.data.load(path)

    def to_sentences(self, paragraph: object) -> tuple[str, ...]:
        extra_abbreviations = self.LANGUAGE_EXTRA_ABREVS.get(self._language, [])
        self._sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
        sentences = self._sentence_tokenizer.tokenize(str(paragraph))
        return tuple(map(str.strip, sentences))

    def to_words(self, sentence: object) -> tuple[str, ...]:
        words = nltk.word_tokenize(str(sentence))
        return tuple(filter(self._is_word, words))

    def _is_word(self, word: str) -> bool:
        return bool(Tokenizer._WORD_PATTERN.search(word))
