
from collections.abc import Iterable
from itertools import chain
from ...utils import cached_property
from ._sentence import Sentence
class Paragraph:
    __slots__ = (
        "_sentences",
        "_cached_property_sentences",
        "_cached_property_headings",
        "_cached_property_words",
    )

    def __init__(self, sentences: Iterable[Sentence]) -> None:
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
        return f"<Paragraph with {len(self.headings)} headings & {len(self.sentences)} sentences>"

    def __repr__(self) -> str:
        return self.__str__()
