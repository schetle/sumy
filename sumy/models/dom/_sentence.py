"""Sentence model representing a single sentence in a document."""

from functools import cached_property


class Sentence:
    """Represents a single sentence in a document.

    Attributes:
        is_heading: Whether this sentence is a heading.
    """

    __slots__ = ("_text", "_tokenizer", "_is_heading", "__dict__")

    def __init__(self, text: str, tokenizer, is_heading: bool = False):
        """Initialize a Sentence.

        Args:
            text: The text content of the sentence.
            tokenizer: Tokenizer instance for word tokenization.
            is_heading: Whether this sentence is a heading.
        """
        self._text = str(text).strip()
        self._tokenizer = tokenizer
        self._is_heading = bool(is_heading)

    @cached_property
    def words(self) -> tuple:
        """Return the words in this sentence.

        Returns:
            Tuple of words extracted by the tokenizer.
        """
        return self._tokenizer.to_words(self._text)

    @property
    def is_heading(self) -> bool:
        """Return whether this sentence is a heading.

        Returns:
            True if the sentence is a heading.
        """
        return self._is_heading

    def __eq__(self, sentence):
        """Check equality with another Sentence.

        Args:
            sentence: Another Sentence to compare with.

        Returns:
            True if both sentences have the same text and heading status.
        """
        assert isinstance(sentence, Sentence)
        return self._is_heading is sentence._is_heading and self._text == sentence._text

    def __ne__(self, sentence):
        """Check inequality with another Sentence.

        Args:
            sentence: Another Sentence to compare with.

        Returns:
            True if the sentences differ.
        """
        return not self.__eq__(sentence)

    def __hash__(self):
        """Return hash of the sentence.

        Returns:
            Hash based on heading status and text.
        """
        return hash((self._is_heading, self._text))

    def __str__(self):
        """Return the text content of the sentence.

        Returns:
            The sentence text.
        """
        return self._text

    def __repr__(self):
        """Return a debug representation of the sentence.

        Returns:
            String showing whether it's a heading or sentence with its text.
        """
        label = "Heading" if self._is_heading else "Sentence"
        return f"<{label}: {self._text}>"
