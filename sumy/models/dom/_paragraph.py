from __future__ import annotations

from itertools import chain
from ...utils import cached_property
from ._sentence import Sentence


class Paragraph(object):
    def __init__(self, sentences) -> None:
        sentences = tuple(sentences)
        for sentence in sentences:
            if not isinstance(sentence, Sentence):
                raise TypeError("Only instances of class 'Sentence' are allowed.")

        self._sentences = sentences

    @cached_property
    def sentences(self) -> tuple[Sentence, ...]:
        return tuple(s for s in self._sentences if not s.is_heading)

    @cached_property
    def headings(self) -> tuple[Sentence, ...]:
        return tuple(s for s in self._sentences if s.is_heading)

    @cached_property
    def words(self) -> tuple[str, ...]:
        return tuple(chain(*(s.words for s in self._sentences)))

    def __str__(self) -> str:
        return "<Paragraph with %d headings & %d sentences>" % (
            len(self.headings),
            len(self.sentences),
        )

    def __repr__(self) -> str:
        return self.__str__()
