"""Paragraph model representing a group of sentences in a document."""

from itertools import chain
from functools import cached_property
from ._sentence import Sentence


class Paragraph:
    """Represents a paragraph containing sentences and headings.

    Attributes:
        sentences: Non-heading sentences in the paragraph.
        headings: Heading sentences in the paragraph.
        words: All words from all sentences in the paragraph.
    """

    __slots__ = ("_sentences", "__dict__")

    def __init__(self, sentences):
        """Initialize a Paragraph with a sequence of sentences.

        Args:
            sentences: Iterable of Sentence instances.

        Raises:
            TypeError: If any element is not a Sentence instance.
        """
        sentences = tuple(sentences)
        for sentence in sentences:
            if not isinstance(sentence, Sentence):
                raise TypeError("Only instances of class 'Sentence' are allowed.")

        self._sentences = sentences

    @cached_property
    def sentences(self) -> tuple:
        """Return non-heading sentences.

        Returns:
            Tuple of sentences that are not headings.
        """
        return tuple(s for s in self._sentences if not s.is_heading)

    @cached_property
    def headings(self) -> tuple:
        """Return heading sentences.

        Returns:
            Tuple of sentences that are headings.
        """
        return tuple(s for s in self._sentences if s.is_heading)

    @cached_property
    def words(self) -> tuple:
        """Return all words from all sentences.

        Returns:
            Tuple of all words from all sentences in the paragraph.
        """
        return tuple(chain(*(s.words for s in self._sentences)))

    def __str__(self):
        """Return a summary string of the paragraph.

        Returns:
            String showing count of headings and sentences.
        """
        return f"<Paragraph with {len(self.headings)} headings & {len(self.sentences)} sentences>"

    def __repr__(self):
        """Return a debug representation of the paragraph.

        Returns:
            Same as __str__.
        """
        return self.__str__()
