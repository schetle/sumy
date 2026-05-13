"""Luhn's heuristic summarization method."""

from ..models import TfDocumentModel
from ._summarizer import AbstractSummarizer


class LuhnSummarizer(AbstractSummarizer):
    """Luhn's heuristic method for automatic text summarization.

    Identifies significant words and rates sentences based on
    clusters of significant words.
    """

    max_gap_size = 4
    # TODO: better recognition of significant words (automatic)
    significant_percentage = 1
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
        """Summarize a document using Luhn's method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        words = self._get_significant_words(document.words)
        return self._get_best_sentences(document.sentences,
            sentences_count, self.rate_sentence, words)

    def _get_significant_words(self, words):
        """Identify significant words by frequency.

        Args:
            words: All words in the document.

        Returns:
            Tuple of significant word stems.
        """
        words = map(self.normalize_word, words)
        words = tuple(self.stem_word(w) for w in words if w not in self._stop_words)

        model = TfDocumentModel(words)

        # take only best `significant_percentage` % words
        best_words_count = int(len(words) * self.significant_percentage)
        words = model.most_frequent_terms(best_words_count)

        # take only words contained multiple times in document
        return tuple(t for t in words if model.term_frequency(t) > 1)

    def rate_sentence(self, sentence, significant_stems):
        """Rate a sentence based on significant word clusters.

        Args:
            sentence: Sentence to rate.
            significant_stems: Tuple of significant word stems.

        Returns:
            Rating score for the sentence.
        """
        ratings = self._get_chunk_ratings(sentence, significant_stems)
        return max(ratings) if ratings else 0

    def _get_chunk_ratings(self, sentence, significant_stems):
        """Compute ratings for word chunks in a sentence.

        Args:
            sentence: Sentence to analyze.
            significant_stems: Significant word stems.

        Returns:
            Tuple of chunk ratings.
        """
        chunks = []
        NONSIGNIFICANT_CHUNK = [0] * self.max_gap_size

        in_chunk = False
        for order, word in enumerate(sentence.words):
            stem = self.stem_word(word)
            # new chunk
            if stem in significant_stems and not in_chunk:
                in_chunk = True
                chunks.append([1])
            # append word to chunk
            elif in_chunk:
                is_significant_word = int(stem in significant_stems)
                chunks[-1].append(is_significant_word)

            # end of chunk
            if chunks and chunks[-1][-self.max_gap_size:] == NONSIGNIFICANT_CHUNK:
                in_chunk = False

        return tuple(map(self._get_chunk_rating, chunks))

    def _get_chunk_rating(self, chunk):
        """Compute the rating for a single chunk of words.

        Args:
            chunk: List of 1s (significant) and 0s (not significant).

        Returns:
            Rating score for the chunk.
        """
        chunk = self._remove_trailing_zeros(chunk)
        words_count = len(chunk)
        assert words_count > 0

        significant_words = sum(chunk)
        if significant_words == 1:
            return 0
        else:
            return significant_words**2 / words_count

    def _remove_trailing_zeros(self, collection):
        """Remove trailing zeroes from an indexable collection of numbers.

        Args:
            collection: List of numbers.

        Returns:
            Collection with trailing zeros removed.
        """
        index = len(collection) - 1
        while index >= 0 and collection[index] == 0:
            index -= 1

        return collection[:index + 1]
