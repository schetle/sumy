from __future__ import annotations

import math

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

from ._summarizer import AbstractSummarizer

if TYPE_CHECKING:
    from ..models.dom import ObjectDocumentModel, Sentence
    from ..utils import ItemsCount


class TextRankSummarizer(AbstractSummarizer):
    """Source: https://github.com/adamfabish/Reduction"""

    def __call__(self, document: ObjectDocumentModel, sentences_count: int | ItemsCount) -> tuple[Sentence, ...]:
        ratings = self.rate_sentences(document)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def rate_sentences(self, document: ObjectDocumentModel) -> dict[Sentence, float]:
        sentences_words = [(s, self._to_words_set(s)) for s in document.sentences]
        ratings: dict[Sentence, float] = defaultdict(float)

        for (sentence1, words1), (sentence2, words2) in combinations(sentences_words, 2):
            rank = self._rate_sentences_edge(words1, words2)
            ratings[sentence1] += rank
            ratings[sentence2] += rank

        return ratings

    def _rate_sentences_edge(self, words1, words2):
        rank = 0
        for w1 in words1:
            for w2 in words2:
                rank += int(w1 == w2)

        if rank == 0:
            return 0.0

        assert len(words1) > 0 and len(words2) > 0
        norm = math.log(len(words1)) + math.log(len(words2))
        return 0.0 if norm == 0.0 else rank / norm
