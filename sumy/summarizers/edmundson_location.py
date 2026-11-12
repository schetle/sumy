# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from itertools import chain
from operator import attrgetter
from typing import Dict, FrozenSet, Tuple, TYPE_CHECKING
from .._compat import ffilter
from ._summarizer import AbstractSummarizer, SentencesCount, Stemmer

if TYPE_CHECKING:
    from ..models.dom import ObjectDocumentModel, Sentence


class EdmundsonLocationMethod(AbstractSummarizer):
    _null_words: FrozenSet[str]

    def __init__(self, stemmer: Stemmer, null_words: FrozenSet[str]) -> None:
        super(EdmundsonLocationMethod, self).__init__(stemmer)
        self._null_words = null_words

    def __call__(self, document: "ObjectDocumentModel", sentences_count: SentencesCount,
            w_h: float, w_p1: float, w_p2: float, w_s1: float, w_s2: float) -> Tuple["Sentence", ...]:
        significant_words = self._compute_significant_words(document)
        ratings = self._rate_sentences(document, significant_words, w_h, w_p1,
            w_p2, w_s1, w_s2)

        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _compute_significant_words(self, document: "ObjectDocumentModel") -> FrozenSet[str]:
        headings = document.headings

        significant_words = chain(*map(attrgetter("words"), headings))
        significant_words = map(self.stem_word, significant_words)
        significant_words = ffilter(self._is_null_word, significant_words)

        return frozenset(significant_words)

    def _is_null_word(self, word: str) -> bool:
        return word in self._null_words

    def _rate_sentences(self, document: "ObjectDocumentModel", significant_words: FrozenSet[str],
            w_h: float, w_p1: float, w_p2: float, w_s1: float, w_s2: float) -> Dict["Sentence", float]:
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

    def _rate_sentence(self, sentence: "Sentence", significant_words: FrozenSet[str]) -> int:
        words = map(self.stem_word, sentence.words)
        return sum(w in significant_words for w in words)

    def rate_sentences(self, document: "ObjectDocumentModel", w_h: float = 1, w_p1: float = 1,
            w_p2: float = 1, w_s1: float = 1, w_s2: float = 1) -> Dict["Sentence", float]:
        significant_words = self._compute_significant_words(document)
        return self._rate_sentences(document, significant_words, w_h, w_p1, w_p2, w_s1, w_s2)
