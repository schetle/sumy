"""Edmundson title method for sentence rating based on heading words."""

from operator import attrgetter
from itertools import chain, filterfalse
from ._summarizer import AbstractSummarizer


class EdmundsonTitleMethod(AbstractSummarizer):
    """Rates sentences based on the presence of words from document headings.

    Words appearing in headings (excluding null words) are considered significant.
    """

    def __init__(self, stemmer, null_words):
        """Initialize the title method.

        Args:
            stemmer: Callable that stems a word.
            null_words: Set of null (stop) word stems to exclude.
        """
        super().__init__(stemmer)
        self._null_words = null_words

    def __call__(self, document, sentences_count):
        """Summarize using the title method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        sentences = document.sentences
        significant_words = self._compute_significant_words(document)

        return self._get_best_sentences(sentences, sentences_count,
            self._rate_sentence, significant_words)

    def _compute_significant_words(self, document):
        """Compute significant words from document headings.

        Args:
            document: ObjectDocumentModel to analyze.

        Returns:
            Frozenset of significant word stems from headings.
        """
        heading_words = map(attrgetter("words"), document.headings)

        significant_words = chain(*heading_words)
        significant_words = map(self.stem_word, significant_words)
        significant_words = filterfalse(self._is_null_word, significant_words)

        return frozenset(significant_words)

    def _is_null_word(self, word):
        """Check if a word is a null (stop) word.

        Args:
            word: Word stem to check.

        Returns:
            True if the word is a null word.
        """
        return word in self._null_words

    def _rate_sentence(self, sentence, significant_words):
        """Rate a sentence by counting significant heading words.

        Args:
            sentence: Sentence to rate.
            significant_words: Frozenset of significant word stems.

        Returns:
            Count of significant words in the sentence.
        """
        words = map(self.stem_word, sentence.words)
        return sum(w in significant_words for w in words)

    def rate_sentences(self, document):
        """Rate all sentences in a document.

        Args:
            document: ObjectDocumentModel to rate.

        Returns:
            Dict mapping sentences to ratings.
        """
        significant_words = self._compute_significant_words(document)

        rated_sentences = {}
        for sentence in document.sentences:
            rated_sentences[sentence] = self._rate_sentence(sentence,
                significant_words)

        return rated_sentences
