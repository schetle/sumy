"""TextRank graph-based summarization method."""

import math

from itertools import combinations
from collections import defaultdict
from ._summarizer import AbstractSummarizer


class TextRankSummarizer(AbstractSummarizer):
    """TextRank summarizer using graph-based ranking of sentences.

    Source: https://github.com/adamfabish/Reduction
    """

    _stop_words = frozenset()

    @property
    def stop_words(self) -> frozenset:
        """Return the current stop words.

        Returns:
            Frozenset of stop words.
        """
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        """Set stop words, normalizing each word.

        Args:
            words: Iterable of stop words.
        """
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        """Summarize a document using TextRank.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        ratings = self.rate_sentences(document)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def rate_sentences(self, document):
        """Rate all sentences in a document using TextRank.

        Args:
            document: ObjectDocumentModel to rate.

        Returns:
            Dict mapping sentences to ratings.
        """
        sentences_words = [(s, self._to_words_set(s)) for s in document.sentences]
        ratings = defaultdict(float)

        for (sentence1, words1), (sentence2, words2) in combinations(sentences_words, 2):
            rank = self._rate_sentences_edge(words1, words2)
            ratings[sentence1] += rank
            ratings[sentence2] += rank

        return ratings

    def _to_words_set(self, sentence):
        """Convert a sentence to a list of stemmed content words.

        Args:
            sentence: Sentence to process.

        Returns:
            List of stemmed word strings.
        """
        words = map(self.normalize_word, sentence.words)
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    def _rate_sentences_edge(self, words1, words2):
        """Compute the edge weight between two sentences.

        Args:
            words1: Word list from first sentence.
            words2: Word list from second sentence.

        Returns:
            Edge weight based on word overlap.
        """
        rank = 0
        for w1 in words1:
            for w2 in words2:
                rank += int(w1 == w2)

        if rank == 0:
            return 0.0

        assert len(words1) > 0 and len(words2) > 0
        norm = math.log(len(words1)) + math.log(len(words2))
        return 0.0 if norm == 0.0 else rank / norm
