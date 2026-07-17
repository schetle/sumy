from collections import namedtuple
from collections.abc import Callable, Sequence
from operator import attrgetter
from typing import Any
from ..utils import ItemsCount
from ..nlp.stemmers import null_stemmer


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class AbstractSummarizer(object):
    def __init__(self, stemmer: Callable[[str], str] = null_stemmer) -> None:
        if not callable(stemmer):
            raise ValueError("Stemmer has to be a callable object")

        self._stemmer = stemmer

    def __call__(self, document: Any, sentences_count: int) -> tuple[Any, ...]:
        raise NotImplementedError("This method should be overriden in subclass")

    def stem_word(self, word: str) -> str:
        return self._stemmer(self.normalize_word(word))

    def normalize_word(self, word: str) -> str:
        return str(word).lower()

    def _get_best_sentences(
        self,
        sentences: Sequence[Any],
        count: int | ItemsCount,
        rating: Callable[..., float] | dict[Any, float],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
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
