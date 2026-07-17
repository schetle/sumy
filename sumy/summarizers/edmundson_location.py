from __future__ import annotations

from itertools import chain, filterfalse
from operator import attrgetter
from typing import TYPE_CHECKING

from ._summarizer import AbstractSummarizer

if TYPE_CHECKING:
    from ..models.dom import ObjectDocumentModel, Sentence
    from ..utils import ItemsCount


class EdmundsonLocationMethod(AbstractSummarizer):
    def __init__(self, stemmer, null_words: frozenset[str]) -> None:
        super(EdmundsonLocationMethod, self).__init__(stemmer)
        self._null_words = null_words

    def __call__(self, document: ObjectDocumentModel, sentences_count: int | ItemsCount, w_h: int, w_p1: int, w_p2: int, w_s1: int, w_s2: int) -> tuple[Sentence, ...]:
        significant_words = self._compute_significant_words(document)
        ratings = self._rate_sentences(document, significant_words, w_h, w_p1,
            w_p2, w_s1, w_s2)

        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _compute_significant_words(self, document):
        headings = document.headings

        significant_words = chain(*map(attrgetter("words"), headings))
        significant_words = map(self.stem_word, significant_words)
        significant_words = filterfalse(self._is_null_word, significant_words)

        return frozenset(significant_words)

    def _is_null_word(self, word):
        return word in self._null_words

    def _rate_sentences(self, document, significant_words, w_h, w_p1, w_p2, w_s1, w_s2):
        rated_sentences = {}
        paragraphs = document.paragraphs

        for paragraph_order, paragraph in enumerate(paragraphs):
            sentences = paragraph.sentences
            for sentence_order, sentence in enumerate(sentences):
                rating = self._rate_sentence(sentence, significant_words)
                rating *= w_h

                if paragraph_order == 0:
                    rating += w_p1
                elif paragraph_order == len(paragraphs) - 1:
                    rating += w_p2

                if sentence_order == 0:
                    rating += w_s1
                elif sentence_order == len(sentences) - 1:
                    rating += w_s2

                rated_sentences[sentence] = rating

        return rated_sentences

    def _rate_sentence(self, sentence, significant_words):
        words = map(self.stem_word, sentence.words)
        return sum(w in significant_words for w in words)

    def rate_sentences(self, document: ObjectDocumentModel, w_h: int = 1, w_p1: int = 1, w_p2: int = 1, w_s1: int = 1, w_s2: int = 1) -> dict[Sentence, float]:
        significant_words = self._compute_significant_words(document)
        return self._rate_sentences(document, significant_words, w_h, w_p1, w_p2, w_s1, w_s2)
