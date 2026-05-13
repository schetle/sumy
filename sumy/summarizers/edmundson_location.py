"""Edmundson location method for sentence rating based on document position."""

from itertools import chain, filterfalse
from operator import attrgetter
from ._summarizer import AbstractSummarizer


class EdmundsonLocationMethod(AbstractSummarizer):
    """Rates sentences based on their position in the document and heading word overlap.

    Considers paragraph position, sentence position within paragraph,
    and overlap with heading words.
    """

    def __init__(self, stemmer, null_words):
        """Initialize the location method.

        Args:
            stemmer: Callable that stems a word.
            null_words: Set of null (stop) word stems to exclude.
        """
        super().__init__(stemmer)
        self._null_words = null_words

    def __call__(self, document, sentences_count, w_h, w_p1, w_p2, w_s1, w_s2):
        """Summarize using the location method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.
            w_h: Heading word weight.
            w_p1: First paragraph weight.
            w_p2: Last paragraph weight.
            w_s1: First sentence weight.
            w_s2: Last sentence weight.

        Returns:
            Tuple of best sentences.
        """
        significant_words = self._compute_significant_words(document)
        ratings = self._rate_sentences(document, significant_words, w_h, w_p1,
            w_p2, w_s1, w_s2)

        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _compute_significant_words(self, document):
        """Compute significant words from document headings.

        Args:
            document: ObjectDocumentModel to analyze.

        Returns:
            Frozenset of significant word stems from headings.
        """
        headings = document.headings

        significant_words = chain(*map(attrgetter("words"), headings))
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

    def _rate_sentences(self, document, significant_words, w_h, w_p1, w_p2, w_s1, w_s2):
        """Rate all sentences based on position and heading word overlap.

        Args:
            document: ObjectDocumentModel to rate.
            significant_words: Frozenset of significant word stems.
            w_h: Heading word weight.
            w_p1: First paragraph weight.
            w_p2: Last paragraph weight.
            w_s1: First sentence weight.
            w_s2: Last sentence weight.

        Returns:
            Dict mapping sentences to ratings.
        """
        rated_sentences = {}
        paragraphs = document.paragraphs

        for paragraph_order, paragraph in enumerate(paragraphs):
            sentences = paragraph.sentences
            for sentence_order, sentence in enumerate(sentences):
                rating = self._rate_sentence(sentence, significant_words)
                rating *= w_h

                if paragraph_order == 0:
                    rating += w_p1
                elif paragraph_order == len(paragraphs) - 1:
                    rating += w_p2

                if sentence_order == 0:
                    rating += w_s1
                elif sentence_order == len(sentences) - 1:
                    rating += w_s2

                rated_sentences[sentence] = rating

        return rated_sentences

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

    def rate_sentences(self, document, w_h=1, w_p1=1, w_p2=1, w_s1=1, w_s2=1):
        """Rate all sentences in a document.

        Args:
            document: ObjectDocumentModel to rate.
            w_h: Heading word weight.
            w_p1: First paragraph weight.
            w_p2: Last paragraph weight.
            w_s1: First sentence weight.
            w_s2: Last sentence weight.

        Returns:
            Dict mapping sentences to ratings.
        """
        significant_words = self._compute_significant_words(document)
        return self._rate_sentences(document, significant_words, w_h, w_p1, w_p2, w_s1, w_s2)
