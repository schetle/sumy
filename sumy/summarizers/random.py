
import random

from ..models.dom import ObjectDocumentModel, Sentence
from ._summarizer import AbstractSummarizer
class RandomSummarizer(AbstractSummarizer):
    """Summarizer that picks sentences randomly."""

    def __call__(self, document: ObjectDocumentModel, sentences_count: int) -> tuple[Sentence, ...]:
        sentences = document.sentences
        ratings = self._get_random_ratings(sentences)

        return self._get_best_sentences(sentences, sentences_count, ratings)

    def _get_random_ratings(self, sentences: tuple[Sentence, ...]) -> dict[Sentence, int]:
        ratings = list(range(len(sentences)))
        random.shuffle(ratings)

        return dict((s, r) for s, r in zip(sentences, ratings))
