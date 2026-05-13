"""Edmundson cue method for sentence rating based on bonus and stigma words."""

from ._summarizer import AbstractSummarizer


class EdmundsonCueMethod(AbstractSummarizer):
    """Rates sentences based on the presence of bonus and stigma words.

    Bonus words increase the rating while stigma words decrease it.
    """

    def __init__(self, stemmer, bonus_words, stigma_words):
        """Initialize the cue method.

        Args:
            stemmer: Callable that stems a word.
            bonus_words: Set of bonus word stems.
            stigma_words: Set of stigma word stems.
        """
        super().__init__(stemmer)
        self._bonus_words = bonus_words
        self._stigma_words = stigma_words

    def __call__(self, document, sentences_count, bunus_word_weight, stigma_word_weight):
        """Summarize using the cue method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.
            bunus_word_weight: Weight multiplier for bonus words.
            stigma_word_weight: Weight multiplier for stigma words.

        Returns:
            Tuple of best sentences.
        """
        return self._get_best_sentences(document.sentences,
            sentences_count, self._rate_sentence, bunus_word_weight,
            stigma_word_weight)

    def _rate_sentence(self, sentence, bunus_word_weight, stigma_word_weight):
        """Rate a sentence based on bonus and stigma word counts.

        Args:
            sentence: Sentence to rate.
            bunus_word_weight: Weight for bonus words.
            stigma_word_weight: Weight for stigma words.

        Returns:
            Rating score (positive bonus minus negative stigma).
        """
        # count number of bonus/stigma words in sentence
        words = map(self.stem_word, sentence.words)
        bonus_words_count, stigma_words_count = self._count_words(words)

        # compute positive & negative rating
        bonus_rating = bonus_words_count * bunus_word_weight
        stigma_rating = stigma_words_count * stigma_word_weight

        # rating of sentence is (positive - negative) rating
        return bonus_rating - stigma_rating

    def _count_words(self, words):
        """Count bonus and stigma words in a word sequence.

        Args:
            words: Iterable of words.

        Returns:
            Tuple of (bonus_words_count, stigma_words_count).
        """
        bonus_words_count = 0
        stigma_words_count = 0

        for word in words:
            if word in self._bonus_words:
                bonus_words_count += 1
            if word in self._stigma_words:
                stigma_words_count += 1

        return bonus_words_count, stigma_words_count

    def rate_sentences(self, document, bunus_word_weight=1, stigma_word_weight=1):
        """Rate all sentences in a document.

        Args:
            document: ObjectDocumentModel to rate.
            bunus_word_weight: Weight for bonus words.
            stigma_word_weight: Weight for stigma words.

        Returns:
            Dict mapping sentences to ratings.
        """
        rated_sentences = {}
        for sentence in document.sentences:
            rated_sentences[sentence] = self._rate_sentence(sentence,
                bunus_word_weight, stigma_word_weight)

        return rated_sentences
