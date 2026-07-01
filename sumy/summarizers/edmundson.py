from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable

from ..models.dom import ObjectDocumentModel, Sentence
from ..nlp.stemmers import null_stemmer
from ._summarizer import AbstractSummarizer
from .edmundson_cue import EdmundsonCueMethod
from .edmundson_key import EdmundsonKeyMethod
from .edmundson_title import EdmundsonTitleMethod
from .edmundson_location import EdmundsonLocationMethod


_EMPTY_SET: frozenset[str] = frozenset()


class EdmundsonSummarizer(AbstractSummarizer):
    _bonus_words: frozenset[str] = _EMPTY_SET
    _stigma_words: frozenset[str] = _EMPTY_SET
    _null_words: frozenset[str] = _EMPTY_SET

    def __init__(self, stemmer: Callable[[str], str] = null_stemmer, cue_weight: float = 1.0,
            key_weight: float = 0.0, title_weight: float = 1.0,
            location_weight: float = 1.0) -> None:
        super().__init__(stemmer)

        self._ensure_correct_weights(cue_weight, key_weight, title_weight,
            location_weight)

        self._cue_weight: float = float(cue_weight)
        self._key_weight: float = float(key_weight)
        self._title_weight: float = float(title_weight)
        self._location_weight: float = float(location_weight)

    def _ensure_correct_weights(self, *weights: float) -> None:
        for w in weights:
            if w < 0.0:
                raise ValueError("Negative wights are not allowed.")

    @property
    def bonus_words(self) -> frozenset[str]:
        return self._bonus_words

    @bonus_words.setter
    def bonus_words(self, collection: Iterable[str]) -> None:
        self._bonus_words = frozenset(map(self.stem_word, collection))

    @property
    def stigma_words(self) -> frozenset[str]:
        return self._stigma_words

    @stigma_words.setter
    def stigma_words(self, collection: Iterable[str]) -> None:
        self._stigma_words = frozenset(map(self.stem_word, collection))

    @property
    def null_words(self) -> frozenset[str]:
        return self._null_words

    @null_words.setter
    def null_words(self, collection: Iterable[str]) -> None:
        self._null_words = frozenset(map(self.stem_word, collection))

    def __call__(self, document: ObjectDocumentModel, sentences_count: int) -> tuple[Sentence, ...]:
        ratings: defaultdict[Sentence, int] = defaultdict(int)

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

    def _update_ratings(self, ratings: defaultdict[Sentence, int],
                        new_ratings: dict[Sentence, float]) -> defaultdict[Sentence, int]:
        assert len(ratings) == 0 or len(ratings) == len(new_ratings)

        for sentence, rating in new_ratings.items():
            ratings[sentence] += rating

        return ratings

    def cue_method(self, document: ObjectDocumentModel, sentences_count: int,
                   bunus_word_value: int = 1, stigma_word_value: int = 1) -> tuple[Sentence, ...]:
        summarization_method = self._build_cue_method_instance()
        return summarization_method(document, sentences_count, bunus_word_value,
            stigma_word_value)

    def _build_cue_method_instance(self) -> EdmundsonCueMethod:
        self.__check_bonus_words()
        self.__check_stigma_words()

        return EdmundsonCueMethod(self._stemmer, self._bonus_words, self._stigma_words)

    def key_method(self, document: ObjectDocumentModel, sentences_count: int,
                   weight: float = 0.5) -> tuple[Sentence, ...]:
        summarization_method = self._build_key_method_instance()
        return summarization_method(document, sentences_count, weight)

    def _build_key_method_instance(self) -> EdmundsonKeyMethod:
        self.__check_bonus_words()

        return  EdmundsonKeyMethod(self._stemmer, self._bonus_words)

    def title_method(self, document: ObjectDocumentModel, sentences_count: int) -> tuple[Sentence, ...]:
        summarization_method = self._build_title_method_instance()
        return summarization_method(document, sentences_count)

    def _build_title_method_instance(self) -> EdmundsonTitleMethod:
        self.__check_null_words()

        return EdmundsonTitleMethod(self._stemmer, self._null_words)

    def location_method(self, document: ObjectDocumentModel, sentences_count: int,
                        w_h: int = 1, w_p1: int = 1, w_p2: int = 1,
                        w_s1: int = 1, w_s2: int = 1) -> tuple[Sentence, ...]:
        summarization_method = self._build_location_method_instance()
        return summarization_method(document, sentences_count, w_h, w_p1, w_p2, w_s1, w_s2)

    def _build_location_method_instance(self) -> EdmundsonLocationMethod:
        self.__check_null_words()

        return EdmundsonLocationMethod(self._stemmer, self._null_words)

    def __check_bonus_words(self) -> None:
        if not self._bonus_words:
            raise ValueError("Set of bonus words is empty. Please set attribute 'bonus_words' with collection of words.")

    def __check_stigma_words(self) -> None:
        if not self._stigma_words:
            raise ValueError("Set of stigma words is empty. Please set attribute 'stigma_words' with collection of words.")

    def __check_null_words(self) -> None:
        if not self._null_words:
            raise ValueError("Set of null words is empty. Please set attribute 'null_words' with collection of words.")
