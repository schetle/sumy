"""Edmundson's heuristic summarization method combining multiple sub-methods."""

from collections import defaultdict
from ..nlp.stemmers import null_stemmer
from ._summarizer import AbstractSummarizer
from .edmundson_cue import EdmundsonCueMethod
from .edmundson_key import EdmundsonKeyMethod
from .edmundson_title import EdmundsonTitleMethod
from .edmundson_location import EdmundsonLocationMethod


_EMPTY_SET = frozenset()


class EdmundsonSummarizer(AbstractSummarizer):
    """Edmundson's method combining cue, key, title, and location heuristics.

    Each sub-method contributes a weighted rating to each sentence.
    """

    _bonus_words = _EMPTY_SET
    _stigma_words = _EMPTY_SET
    _null_words = _EMPTY_SET

    def __init__(self, stemmer=null_stemmer, cue_weight=1.0, key_weight=0.0,
            title_weight=1.0, location_weight=1.0):
        """Initialize an EdmundsonSummarizer with sub-method weights.

        Args:
            stemmer: Callable that stems a word.
            cue_weight: Weight for the cue method.
            key_weight: Weight for the key method.
            title_weight: Weight for the title method.
            location_weight: Weight for the location method.

        Raises:
            ValueError: If any weight is negative.
        """
        super().__init__(stemmer)

        self._ensure_correct_weights(cue_weight, key_weight, title_weight,
            location_weight)

        self._cue_weight = float(cue_weight)
        self._key_weight = float(key_weight)
        self._title_weight = float(title_weight)
        self._location_weight = float(location_weight)

    def _ensure_correct_weights(self, *weights):
        """Validate that all weights are non-negative.

        Args:
            *weights: Weight values to validate.

        Raises:
            ValueError: If any weight is negative.
        """
        for w in weights:
            if w < 0.0:
                raise ValueError("Negative weights are not allowed.")

    @property
    def bonus_words(self) -> frozenset:
        """Return the current bonus words.

        Returns:
            Frozenset of bonus words.
        """
        return self._bonus_words

    @bonus_words.setter
    def bonus_words(self, collection):
        """Set bonus words, stemming each word.

        Args:
            collection: Iterable of bonus words.
        """
        self._bonus_words = frozenset(map(self.stem_word, collection))

    @property
    def stigma_words(self) -> frozenset:
        """Return the current stigma words.

        Returns:
            Frozenset of stigma words.
        """
        return self._stigma_words

    @stigma_words.setter
    def stigma_words(self, collection):
        """Set stigma words, stemming each word.

        Args:
            collection: Iterable of stigma words.
        """
        self._stigma_words = frozenset(map(self.stem_word, collection))

    @property
    def null_words(self) -> frozenset:
        """Return the current null words.

        Returns:
            Frozenset of null words.
        """
        return self._null_words

    @null_words.setter
    def null_words(self, collection):
        """Set null words, stemming each word.

        Args:
            collection: Iterable of null words.
        """
        self._null_words = frozenset(map(self.stem_word, collection))

    def __call__(self, document, sentences_count):
        """Summarize a document using Edmundson's combined method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        ratings = defaultdict(int)

        if self._cue_weight > 0.0:
            method = self._build_cue_method_instance()
            ratings = self._update_ratings(ratings, method.rate_sentences(document))
        if self._key_weight > 0.0:
            method = self._build_key_method_instance()
            ratings = self._update_ratings(ratings, method.rate_sentences(document))
        if self._title_weight > 0.0:
            method = self._build_title_method_instance()
            ratings = self._update_ratings(ratings, method.rate_sentences(document))
        if self._location_weight > 0.0:
            method = self._build_location_method_instance()
            ratings = self._update_ratings(ratings, method.rate_sentences(document))

        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _update_ratings(self, ratings, new_ratings):
        """Merge new ratings into existing ratings.

        Args:
            ratings: Existing sentence ratings dict.
            new_ratings: New ratings to add.

        Returns:
            Updated ratings dict.
        """
        assert len(ratings) == 0 or len(ratings) == len(new_ratings)

        for sentence, rating in new_ratings.items():
            ratings[sentence] += rating

        return ratings

    def cue_method(self, document, sentences_count, bunus_word_value=1, stigma_word_value=1):
        """Run only the cue sub-method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.
            bunus_word_value: Weight for bonus words.
            stigma_word_value: Weight for stigma words.

        Returns:
            Tuple of best sentences.
        """
        summarization_method = self._build_cue_method_instance()
        return summarization_method(document, sentences_count, bunus_word_value,
            stigma_word_value)

    def _build_cue_method_instance(self):
        """Build an EdmundsonCueMethod instance.

        Returns:
            EdmundsonCueMethod instance.

        Raises:
            ValueError: If bonus or stigma words are not set.
        """
        self.__check_bonus_words()
        self.__check_stigma_words()

        return EdmundsonCueMethod(self._stemmer, self._bonus_words, self._stigma_words)

    def key_method(self, document, sentences_count, weight=0.5):
        """Run only the key sub-method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.
            weight: Minimum frequency weight for significant words.

        Returns:
            Tuple of best sentences.
        """
        summarization_method = self._build_key_method_instance()
        return summarization_method(document, sentences_count, weight)

    def _build_key_method_instance(self):
        """Build an EdmundsonKeyMethod instance.

        Returns:
            EdmundsonKeyMethod instance.

        Raises:
            ValueError: If bonus words are not set.
        """
        self.__check_bonus_words()

        return EdmundsonKeyMethod(self._stemmer, self._bonus_words)

    def title_method(self, document, sentences_count):
        """Run only the title sub-method.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        summarization_method = self._build_title_method_instance()
        return summarization_method(document, sentences_count)

    def _build_title_method_instance(self):
        """Build an EdmundsonTitleMethod instance.

        Returns:
            EdmundsonTitleMethod instance.

        Raises:
            ValueError: If null words are not set.
        """
        self.__check_null_words()

        return EdmundsonTitleMethod(self._stemmer, self._null_words)

    def location_method(self, document, sentences_count, w_h=1, w_p1=1, w_p2=1, w_s1=1, w_s2=1):
        """Run only the location sub-method.

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
        summarization_method = self._build_location_method_instance()
        return summarization_method(document, sentences_count, w_h, w_p1, w_p2, w_s1, w_s2)

    def _build_location_method_instance(self):
        """Build an EdmundsonLocationMethod instance.

        Returns:
            EdmundsonLocationMethod instance.

        Raises:
            ValueError: If null words are not set.
        """
        self.__check_null_words()

        return EdmundsonLocationMethod(self._stemmer, self._null_words)

    def __check_bonus_words(self):
        """Verify that bonus words have been set.

        Raises:
            ValueError: If bonus words are empty.
        """
        if not self._bonus_words:
            raise ValueError("Set of bonus words is empty. Please set attribute 'bonus_words' with collection of words.")

    def __check_stigma_words(self):
        """Verify that stigma words have been set.

        Raises:
            ValueError: If stigma words are empty.
        """
        if not self._stigma_words:
            raise ValueError("Set of stigma words is empty. Please set attribute 'stigma_words' with collection of words.")

    def __check_null_words(self):
        """Verify that null words have been set.

        Raises:
            ValueError: If null words are empty.
        """
        if not self._null_words:
            raise ValueError("Set of null words is empty. Please set attribute 'null_words' with collection of words.")
