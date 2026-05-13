"""Term-Frequency document model for text analysis."""

import math

from pprint import pformat
from collections import Counter
from collections.abc import Sequence


class TfDocumentModel:
    """Term-Frequency document model (term = word).

    Provides term frequency statistics for a collection of words.
    """

    def __init__(self, words, tokenizer=None):
        """Initialize TfDocumentModel with words or text.

        Args:
            words: A sequence of words or a string (requires tokenizer).
            tokenizer: Tokenizer to split string into words. Required if words is a string.

        Raises:
            ValueError: If words is a string without tokenizer, or not a valid sequence.
        """
        if isinstance(words, str) and tokenizer is None:
            raise ValueError(
                "Tokenizer has to be given if ``words`` is not a sequence.")
        elif isinstance(words, str):
            words = tokenizer.to_words(words)
        elif not isinstance(words, Sequence):
            raise ValueError(
                "Parameter ``words`` has to be sequence or string with tokenizer given.")

        self._terms = Counter(w.lower() for w in words)
        self._max_frequency = max(self._terms.values()) if self._terms else 1

    @property
    def magnitude(self) -> float:
        """Return the length/norm/magnitude of the vector representation.

        Returns:
            The L2 norm of the term frequency vector.
        """
        return math.sqrt(sum(t**2 for t in self._terms.values()))

    @property
    def terms(self):
        """Return the terms in the document.

        Returns:
            View of the term keys.
        """
        return self._terms.keys()

    def most_frequent_terms(self, count: int = 0) -> tuple:
        """Return terms sorted by frequency in descending order.

        Args:
            count: Max number of returned terms. 0 means no limit.

        Returns:
            Tuple of terms sorted by frequency.

        Raises:
            ValueError: If count is negative.
        """
        # sort terms by number of occurrences in descending order
        terms = sorted(self._terms.items(), key=lambda i: -i[1])

        terms = tuple(i[0] for i in terms)
        if count == 0:
            return terms
        elif count > 0:
            return terms[:count]
        else:
            raise ValueError(
                "Only non-negative values are allowed for count of terms.")

    def term_frequency(self, term: str) -> int:
        """Return frequency of a term in the document.

        Args:
            term: The term to look up.

        Returns:
            Count of the term in the document.
        """
        return self._terms.get(term, 0)

    def normalized_term_frequency(self, term: str, smooth: float = 0.0) -> float:
        """Return normalized frequency of a term in the document.

        See: http://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html

        Args:
            term: The term to look up.
            smooth: Smoothing parameter (0.0 <= smooth <= 1.0). Defaults to 0.0.

        Returns:
            Normalized frequency between 0.0 and 1.0.
        """
        frequency = self.term_frequency(term) / self._max_frequency
        return smooth + (1.0 - smooth) * frequency

    def __repr__(self):
        """Return a debug representation of the model.

        Returns:
            String showing the term frequencies.
        """
        return f"<TfDocumentModel {pformat(self._terms)}>"
