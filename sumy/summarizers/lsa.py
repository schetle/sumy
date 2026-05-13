"""Latent Semantic Analysis (LSA) summarization method."""

import math

from warnings import warn

import numpy
from numpy.linalg import svd as singular_value_decomposition
from ._summarizer import AbstractSummarizer


class LsaSummarizer(AbstractSummarizer):
    """LSA-based summarizer using Singular Value Decomposition.

    Decomposes the term-sentence matrix using SVD and ranks sentences
    based on their contribution to the most important topics.
    """

    MIN_DIMENSIONS = 3
    REDUCTION_RATIO = 1 / 1
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
        """Summarize a document using LSA.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        dictionary = self._create_dictionary(document)
        # empty document
        if not dictionary:
            return ()

        matrix = self._create_matrix(document, dictionary)
        matrix = self._compute_term_frequency(matrix)
        u, sigma, v = singular_value_decomposition(matrix, full_matrices=False)

        ranks = iter(self._compute_ranks(sigma, v))
        return self._get_best_sentences(document.sentences, sentences_count,
            lambda s: next(ranks))

    def _create_dictionary(self, document):
        """Create a mapping of words to row indices.

        Args:
            document: ObjectDocumentModel to analyze.

        Returns:
            Dict mapping word stems to row indices.
        """
        words = map(self.normalize_word, document.words)
        unique_words = frozenset(self.stem_word(w) for w in words if w not in self._stop_words)

        return dict((w, i) for i, w in enumerate(unique_words))

    def _create_matrix(self, document, dictionary):
        """Create a term-sentence occurrence matrix.

        Args:
            document: ObjectDocumentModel to analyze.
            dictionary: Word-to-index mapping.

        Returns:
            NumPy matrix of shape |unique words| x |sentences|.
        """
        sentences = document.sentences

        words_count = len(dictionary)
        sentences_count = len(sentences)
        if words_count < sentences_count:
            message = (
                "Number of words (%d) is lower than number of sentences (%d). "
                "LSA algorithm may not work properly."
            )
            warn(message % (words_count, sentences_count))

        # create matrix |unique words|×|sentences| filled with zeroes
        matrix = numpy.zeros((words_count, sentences_count))
        for col, sentence in enumerate(sentences):
            for word in map(self.stem_word, sentence.words):
                # only valid words are counted (not stop-words, ...)
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, col] += 1

        return matrix

    def _compute_term_frequency(self, matrix, smooth=0.4):
        """Compute TF metrics for each sentence column in the matrix.

        See: http://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html

        Args:
            matrix: Term-sentence matrix.
            smooth: Smoothing parameter (0.0 <= smooth < 1.0).

        Returns:
            Matrix with normalized term frequencies.
        """
        assert 0.0 <= smooth < 1.0

        max_word_frequencies = numpy.max(matrix, axis=0)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                max_word_frequency = max_word_frequencies[col]
                if max_word_frequency != 0:
                    frequency = matrix[row, col] / max_word_frequency
                    matrix[row, col] = smooth + (1.0 - smooth) * frequency

        return matrix

    def _compute_ranks(self, sigma, v_matrix):
        """Compute sentence ranks from SVD results.

        Args:
            sigma: Singular values from SVD.
            v_matrix: Right singular vectors from SVD.

        Returns:
            List of sentence rank scores.
        """
        assert len(sigma) == v_matrix.shape[0], "Matrices should be multiplicable"

        dimensions = max(LsaSummarizer.MIN_DIMENSIONS,
            int(len(sigma) * LsaSummarizer.REDUCTION_RATIO))
        powered_sigma = tuple(s**2 if i < dimensions else 0.0
            for i, s in enumerate(sigma))

        ranks = []
        # iterate over columns of matrix (rows of transposed matrix)
        for column_vector in v_matrix.T:
            rank = sum(s * v**2 for s, v in zip(powered_sigma, column_vector))
            ranks.append(math.sqrt(rank))

        return ranks
