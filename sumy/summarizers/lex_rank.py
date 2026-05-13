"""LexRank graph-based summarization method."""

import math
from collections import Counter

import numpy
from ._summarizer import AbstractSummarizer


class LexRankSummarizer(AbstractSummarizer):
    """LexRank: Graph-based Centrality as Salience in Text Summarization.

    Source: http://tangra.si.umich.edu/~radev/lexrank/lexrank.pdf

    Uses TF-IDF cosine similarity to build a sentence graph and
    applies the power method to compute centrality scores.
    """

    threshold = 0.1
    epsilon = 0.1
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
        """Summarize a document using LexRank.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        sentences_words = [self._to_words_set(s) for s in document.sentences]
        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)

        matrix = self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(document.sentences, scores))

        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _to_words_set(self, sentence):
        """Convert a sentence to a list of stemmed content words.

        Args:
            sentence: Sentence to process.

        Returns:
            List of stemmed word strings.
        """
        words = map(self.normalize_word, sentence.words)
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    def _compute_tf(self, sentences):
        """Compute normalized term frequency for each sentence.

        Args:
            sentences: List of word lists.

        Returns:
            List of dicts mapping terms to TF values.
        """
        tf_values = map(Counter, sentences)

        tf_metrics = []
        for sentence in tf_values:
            metrics = {}
            max_tf = self._find_tf_max(sentence)

            for term, tf in sentence.items():
                metrics[term] = tf / max_tf

            tf_metrics.append(metrics)

        return tf_metrics

    @staticmethod
    def _find_tf_max(terms):
        """Find the maximum term frequency in a counter.

        Args:
            terms: Counter of term frequencies.

        Returns:
            Maximum frequency value, or 1 if empty.
        """
        return max(terms.values()) if terms else 1

    @staticmethod
    def _compute_idf(sentences):
        """Compute inverse document frequency for each term.

        Args:
            sentences: List of word lists.

        Returns:
            Dict mapping terms to IDF values.
        """
        idf_metrics = {}
        sentences_count = len(sentences)

        for sentence in sentences:
            for term in sentence:
                if term not in idf_metrics:
                    n_j = sum(1 for s in sentences if term in s)
                    idf_metrics[term] = math.log(sentences_count / (1 + n_j))

        return idf_metrics

    def _create_matrix(self, sentences, threshold, tf_metrics, idf_metrics):
        """Create the sentence adjacency matrix.

        Args:
            sentences: List of word lists.
            threshold: Cosine similarity threshold for edge creation.
            tf_metrics: TF values per sentence.
            idf_metrics: IDF values per term.

        Returns:
            NumPy matrix of shape |sentences| x |sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count,))

        for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
            for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
                matrix[row, col] = self._compute_cosine(sentence1, sentence2, tf1, tf2, idf_metrics)

                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row, col] = 0

        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix

    @staticmethod
    def _compute_cosine(sentence1, sentence2, tf1, tf2, idf_metrics):
        """Compute TF-IDF weighted cosine similarity between two sentences.

        Args:
            sentence1: First sentence word list.
            sentence2: Second sentence word list.
            tf1: TF values for first sentence.
            tf2: TF values for second sentence.
            idf_metrics: IDF values for all terms.

        Returns:
            Cosine similarity score.
        """
        common_words = frozenset(sentence1) & frozenset(sentence2)

        numerator = 0.0
        for term in common_words:
            numerator += tf1[term] * tf2[term] * idf_metrics[term]**2

        denominator1 = sum((tf1[t] * idf_metrics[t])**2 for t in sentence1)
        denominator2 = sum((tf2[t] * idf_metrics[t])**2 for t in sentence2)

        if denominator1 > 0 and denominator2 > 0:
            return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            return 0.0

    @staticmethod
    def power_method(matrix, epsilon):
        """Apply the power method to compute stationary distribution.

        Args:
            matrix: Transition matrix.
            epsilon: Convergence threshold.

        Returns:
            NumPy array of sentence scores.
        """
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector
