from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from ..models.dom import ObjectDocumentModel, Sentence
from ._summarizer import AbstractSummarizer


class EdmundsonKeyMethod(AbstractSummarizer):
    def __init__(self, stemmer: Callable[[str], str], bonus_words: frozenset[str]) -> None:
        super().__init__(stemmer)
        self._bonus_words: frozenset[str] = bonus_words

    def __call__(self, document: ObjectDocumentModel, sentences_count: int,
                 weight: float = 0.5) -> tuple[Sentence, ...]:
        significant_words = self._compute_significant_words(document, weight)

        return self._get_best_sentences(document.sentences,
            sentences_count, self._rate_sentence, significant_words)

    def _compute_significant_words(self, document: ObjectDocumentModel,
                                   weight: float) -> tuple[str, ...]:
        # keep only stems contained in bonus words
        words = map(self.stem_word, document.words)
        words = filter(self._is_bonus_word, words)

        # compute frequencies of bonus words in document
        word_counts = Counter(words)
        word_frequencies = word_counts.values()

        # no frequencies means no significant words
        if not word_frequencies:
            return ()

        # return only words greater than weight
        max_word_frequency = max(word_frequencies)
        return tuple(word for word, frequency in word_counts.items()
            if frequency/max_word_frequency > weight)

    def _is_bonus_word(self, word: str) -> bool:
        return word in self._bonus_words

    def _rate_sentence(self, sentence: Sentence, significant_words: tuple[str, ...]) -> int:
        words = map(self.stem_word, sentence.words)
        return sum(w in significant_words for w in words)

    def rate_sentences(self, document: ObjectDocumentModel, weight: float = 0.5) -> dict[Sentence, int]:
        significant_words = self._compute_significant_words(document, weight)

        rated_sentences: dict[Sentence, int] = {}
        for sentence in document.sentences:
            rated_sentences[sentence] = self._rate_sentence(sentence,
                significant_words)

        return rated_sentences
