# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from typing import Dict, FrozenSet, Tuple, TYPE_CHECKING
from .._compat import Counter
from ._summarizer import AbstractSummarizer, SentencesCount, Stemmer

if TYPE_CHECKING:
    from ..models.dom import ObjectDocumentModel, Sentence


class EdmundsonKeyMethod(AbstractSummarizer):
    _bonus_words: FrozenSet[str]

    def __init__(self, stemmer: Stemmer, bonus_words: FrozenSet[str]) -> None:
        super(EdmundsonKeyMethod, self).__init__(stemmer)
        self._bonus_words = bonus_words

    def __call__(self, document: "ObjectDocumentModel", sentences_count: SentencesCount, weight: float) -> Tuple["Sentence", ...]:
        significant_words = self._compute_significant_words(document, weight)

        return self._get_best_sentences(document.sentences,
            sentences_count, self._rate_sentence, significant_words)

    def _compute_significant_words(self, document: "ObjectDocumentModel", weight: float) -> Tuple[str, ...]:
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

    def _rate_sentence(self, sentence: "Sentence", significant_words: Tuple[str, ...]) -> int:
        words = map(self.stem_word, sentence.words)
        return sum(w in significant_words for w in words)

    def rate_sentences(self, document: "ObjectDocumentModel", weight: float = 0.5) -> Dict["Sentence", float]:
        significant_words = self._compute_significant_words(document, weight)

        rated_sentences = {}
        for sentence in document.sentences:
            rated_sentences[sentence] = self._rate_sentence(sentence,
                significant_words)

        return rated_sentences
