"""Edmundson key method for sentence rating based on keyword frequency."""

from collections import Counter
from ._summarizer import AbstractSummarizer


class EdmundsonKeyMethod(AbstractSummarizer):
    """Rates sentences based on the frequency of bonus keywords in the document.

    Words that appear frequently and are in the bonus word set are considered significant.
    """

    def __init__(self, stemmer, bonus_words):
        """Initialize the key method.

        Args:
            stemmer: Callable that stems a word.
            bonus_words: Set of bonus word stems.
        """
        super().__init__(stemmer)
        self._bonus_words = bonus_words

    def __call__(self, document, sentences_count, weight):
        """Summarize using the key method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.
            weight: Minimum frequency ratio for significant words.

        Returns:
            Tuple of best sentences.
        """
        significant_words = self._compute_significant_words(document, weight)

        return self._get_best_sentences(document.sentences,
            sentences_count, self._rate_sentence, significant_words)

    def _compute_significant_words(self, document, weight):
        """Compute significant words based on frequency above a weight threshold.

        Args:
            document: ObjectDocumentModel to analyze.
            weight: Minimum frequency ratio (0.0 to 1.0).

        Returns:
            Tuple of significant word stems.
        """
        # keep only stems contained in bonus words
        words = map(self.stem_word, document.words)
        words = filter(self._is_bonus_word, words)

        # compute frequencies of bonus words in document
        word_counts = Counter(words)
        word_frequencies = word_counts.values()

        # no frequencies means no significant words
        if not word_frequencies:
            return ()

        # return only words greater than weight
        max_word_frequency = max(word_frequencies)
        return tuple(word for word, frequency in word_counts.items()
            if frequency / max_word_frequency > weight)

    def _is_bonus_word(self, word):
        """Check if a word is in the bonus word set.

        Args:
            word: Word stem to check.

        Returns:
            True if the word is a bonus word.
        """
        return word in self._bonus_words

    def _rate_sentence(self, sentence, significant_words):
        """Rate a sentence by counting significant words.

        Args:
            sentence: Sentence to rate.
            significant_words: Tuple of significant word stems.

        Returns:
            Count of significant words in the sentence.
        """
        words = map(self.stem_word, sentence.words)
        return sum(w in significant_words for w in words)

    def rate_sentences(self, document, weight=0.5):
        """Rate all sentences in a document.

        Args:
            document: ObjectDocumentModel to rate.
            weight: Minimum frequency ratio for significant words.

        Returns:
            Dict mapping sentences to ratings.
        """
        significant_words = self._compute_significant_words(document, weight)

        rated_sentences = {}
        for sentence in document.sentences:
            rated_sentences[sentence] = self._rate_sentence(sentence,
                significant_words)

        return rated_sentences
