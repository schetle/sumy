from itertools import chain
from ...utils import cached_property
from ._paragraph import Paragraph
from ._sentence import Sentence


class ObjectDocumentModel(object):
    def __init__(self, paragraphs) -> None:
        self._paragraphs = tuple(paragraphs)

    @property
    def paragraphs(self) -> tuple[Paragraph, ...]:
        return self._paragraphs

    @cached_property
    def sentences(self) -> tuple[Sentence, ...]:
        sentences = (p.sentences for p in self._paragraphs)
        return tuple(chain(*sentences))

    @cached_property
    def headings(self) -> tuple[Sentence, ...]:
        headings = (p.headings for p in self._paragraphs)
        return tuple(chain(*headings))

    @cached_property
    def words(self) -> tuple[str, ...]:
        words = (p.words for p in self._paragraphs)
        return tuple(chain(*words))

    def __str__(self) -> str:
        return "<DOM with %d paragraphs>" % len(self.paragraphs)

    def __repr__(self) -> str:
        return self.__str__()
