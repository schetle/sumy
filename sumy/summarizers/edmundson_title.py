from __future__ import annotations

from collections.abc import Callable
from itertools import chain, filterfalse
from operator import attrgetter

from ..models.dom import ObjectDocumentModel, Sentence
from ._summarizer import AbstractSummarizer


class EdmundsonTitleMethod(AbstractSummarizer):
    def __init__(self, stemmer: Callable[[str], str], null_words: frozenset[str]) -> None:
        super().__init__(stemmer)
        self._null_words: frozenset[str] = null_words

    def __call__(self, document: ObjectDocumentModel, sentences_count: int) -> tuple[Sentence, ...]:
        sentences = document.sentences
        significant_words = self._compute_significant_words(document)

        return self._get_best_sentences(sentences, sentences_count,
            self._rate_sentence, significant_words)

    def _compute_significant_words(self, document: ObjectDocumentModel) -> frozenset[str]:
        heading_words = map(attrgetter("words"), document.headings)

        significant_words = chain(*heading_words)
        significant_words = map(self.stem_word, significant_words)
        significant_words = filterfalse(self._is_null_word, significant_words)

        return frozenset(significant_words)

    def _is_null_word(self, word: str) -> bool:
        return word in self._null_words

    def _rate_sentence(self, sentence: Sentence, significant_words: frozenset[str]) -> int:
        words = map(self.stem_word, sentence.words)
        return sum(w in significant_words for w in words)

    def rate_sentences(self, document: ObjectDocumentModel) -> dict[Sentence, int]:
        significant_words = self._compute_significant_words(document)

        rated_sentences: dict[Sentence, int] = {}
        for sentence in document.sentences:
            rated_sentences[sentence] = self._rate_sentence(sentence,
                significant_words)

        return rated_sentences
