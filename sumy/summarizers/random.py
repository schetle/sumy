"""Random baseline summarization method."""

import random

from ._summarizer import AbstractSummarizer


class RandomSummarizer(AbstractSummarizer):
    """Summarizer that picks sentences randomly.

    Useful as a baseline for evaluating other summarizers.
    """

    def __call__(self, document, sentences_count):
        """Summarize a document by selecting random sentences.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of randomly selected sentences.
        """
        sentences = document.sentences
        ratings = self._get_random_ratings(sentences)

        return self._get_best_sentences(sentences, sentences_count, ratings)

    def _get_random_ratings(self, sentences):
        """Assign random ratings to sentences.

        Args:
            sentences: Tuple of Sentence objects.

        Returns:
            Dict mapping sentences to random ratings.
        """
        ratings = list(range(len(sentences)))
        random.shuffle(ratings)

        return dict((s, r) for s, r in zip(sentences, ratings))
