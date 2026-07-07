from collections import namedtuple
from collections.abc import Iterable
from operator import attrgetter
from typing import TYPE_CHECKING, Callable, Union

from ..utils import ItemsCount
from ..nlp.stemmers import null_stemmer

if TYPE_CHECKING:
    from ..models.dom import ObjectDocumentModel, Sentence


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class AbstractSummarizer(object):
    def __init__(self, stemmer: Callable = null_stemmer) -> None:
        if not callable(stemmer):
            raise ValueError("Stemmer has to be a callable object")

        self._stemmer = stemmer

    def __call__(self, document: "ObjectDocumentModel", sentences_count: Union[int, str, ItemsCount]) -> tuple:
        raise NotImplementedError("This method should be overriden in subclass")

    def stem_word(self, word: str) -> str:
        return str(self._stemmer(self.normalize_word(word)))

    def normalize_word(self, word: str) -> str:
        return null_stemmer(word)

    def _get_best_sentences(self, sentences: 'Iterable[Sentence]', count: Union[int, ItemsCount], rating: Union[dict, Callable], *args, **kwargs) -> tuple:
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos: list[SentenceInfo] = [SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences)]

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        if not isinstance(count, ItemsCount):
            count = ItemsCount(count)
        infos = count(infos)
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos)
