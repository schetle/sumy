# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals


from collections import namedtuple
from operator import attrgetter
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING, Tuple, Union
from ..utils import ItemsCount
from .._compat import to_unicode
from ..nlp.stemmers import null_stemmer

if TYPE_CHECKING:
    from ..models.dom import ObjectDocumentModel, Sentence


# stemmer is any callable converting a (normalized) word to its stem
Stemmer = Callable[[str], str]
# number of sentences to select: absolute count, percentage string (e.g. "20%") or ItemsCount
SentencesCount = Union[int, str, ItemsCount]


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class AbstractSummarizer(object):
    _stemmer: Stemmer

    def __init__(self, stemmer: Stemmer = null_stemmer) -> None:
        if not callable(stemmer):
            raise ValueError("Stemmer has to be a callable object")

        self._stemmer = stemmer

    def __call__(self, document: "ObjectDocumentModel", sentences_count: SentencesCount) -> Tuple["Sentence", ...]:
        raise NotImplementedError("This method should be overriden in subclass")

    def stem_word(self, word: str) -> str:
        return self._stemmer(self.normalize_word(word))

    def normalize_word(self, word: str) -> str:
        return to_unicode(word).lower()

    def _get_best_sentences(
            self,
            sentences: Iterable["Sentence"],
            count: SentencesCount,
            rating: Union[Mapping["Sentence", float], Callable[..., float]],
            *args: Any,
            **kwargs: Any
    ) -> Tuple["Sentence", ...]:
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        if not isinstance(count, ItemsCount):
            count = ItemsCount(count)
        infos = count(infos)
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos)
