"""Document model representing a parsed document as paragraphs, sentences, and words."""

from itertools import chain
from functools import cached_property


class ObjectDocumentModel:
    """Represents a parsed document as a hierarchy of paragraphs, sentences, and words.

    This is the core data model that all summarizers operate on.
    """

    def __init__(self, paragraphs):
        """Initialize an ObjectDocumentModel with paragraphs.

        Args:
            paragraphs: Iterable of Paragraph instances.
        """
        self._paragraphs = tuple(paragraphs)

    @property
    def paragraphs(self) -> tuple:
        """Return the paragraphs in the document.

        Returns:
            Tuple of Paragraph instances.
        """
        return self._paragraphs

    @cached_property
    def sentences(self) -> tuple:
        """Return all non-heading sentences from all paragraphs.

        Returns:
            Tuple of Sentence instances.
        """
        sentences = (p.sentences for p in self._paragraphs)
        return tuple(chain(*sentences))

    @cached_property
    def headings(self) -> tuple:
        """Return all heading sentences from all paragraphs.

        Returns:
            Tuple of heading Sentence instances.
        """
        headings = (p.headings for p in self._paragraphs)
        return tuple(chain(*headings))

    @cached_property
    def words(self) -> tuple:
        """Return all words from all paragraphs.

        Returns:
            Tuple of all words in the document.
        """
        words = (p.words for p in self._paragraphs)
        return tuple(chain(*words))

    def __str__(self):
        """Return a summary string of the document.

        Returns:
            String showing the number of paragraphs.
        """
        return f"<DOM with {len(self.paragraphs)} paragraphs>"

    def __repr__(self):
        """Return a debug representation of the document.

        Returns:
            Same as __str__.
        """
        return self.__str__()
