from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ._summarizer import AbstractSummarizer

if TYPE_CHECKING:
    from ..models.dom import ObjectDocumentModel, Sentence
    from ..utils import ItemsCount


class RandomSummarizer(AbstractSummarizer):
    """Summarizer that picks sentences randomly."""

    def __call__(self, document: ObjectDocumentModel, sentences_count: int | ItemsCount) -> tuple[Sentence, ...]:
        sentences = document.sentences
        ratings = self._get_random_ratings(sentences)

        return self._get_best_sentences(sentences, sentences_count, ratings)

    def _get_random_ratings(self, sentences):
        ratings = list(range(len(sentences)))
        random.shuffle(ratings)

        return dict((s, r) for s, r in zip(sentences, ratings))
