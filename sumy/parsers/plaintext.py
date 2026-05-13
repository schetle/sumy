"""Parser for plain text documents."""

from functools import cached_property
from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from .parser import DocumentParser


class PlaintextParser(DocumentParser):
    """Parser of text from plaintext format into DOM.

    Handles paragraph detection (blank lines), heading detection (all-caps lines),
    and sentence splitting.
    """

    @classmethod
    def from_string(cls, string: str, tokenizer) -> "PlaintextParser":
        """Create a parser from a string.

        Args:
            string: Plain text content.
            tokenizer: Tokenizer instance.

        Returns:
            PlaintextParser instance.
        """
        return cls(string, tokenizer)

    @classmethod
    def from_file(cls, file_path: str, tokenizer) -> "PlaintextParser":
        """Create a parser from a file.

        Args:
            file_path: Path to the text file.
            tokenizer: Tokenizer instance.

        Returns:
            PlaintextParser instance.
        """
        with open(file_path) as file:
            return cls(file.read(), tokenizer)

    def __init__(self, text, tokenizer):
        """Initialize a PlaintextParser.

        Args:
            text: Plain text content (string or bytes).
            tokenizer: Tokenizer instance.
        """
        super().__init__(tokenizer)
        if isinstance(text, bytes):
            text = text.decode("utf8")
        self._text = str(text).strip()

    @cached_property
    def significant_words(self) -> tuple:
        """Return significant words extracted from headings.

        Returns:
            Tuple of words from headings, or default significant words.
        """
        words = []
        for paragraph in self.document.paragraphs:
            for heading in paragraph.headings:
                words.extend(heading.words)

        if words:
            return tuple(words)
        else:
            return self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self) -> tuple:
        """Return stigma words.

        Returns:
            Default stigma words (plaintext has no link/strikethrough markup).
        """
        return self.STIGMA_WORDS

    @cached_property
    def document(self) -> ObjectDocumentModel:
        """Parse the text into an ObjectDocumentModel.

        Returns:
            ObjectDocumentModel with paragraphs, sentences, and words.
        """
        current_paragraph = []
        paragraphs = []
        for line in self._text.splitlines():
            line = line.strip()
            if line.isupper():
                heading = Sentence(line, self._tokenizer, is_heading=True)
                current_paragraph.append(heading)
            elif not line and current_paragraph:
                sentences = self._to_sentences(current_paragraph)
                paragraphs.append(Paragraph(sentences))
                current_paragraph = []
            elif line:
                current_paragraph.append(line)

        sentences = self._to_sentences(current_paragraph)
        paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)

    def _to_sentences(self, lines):
        """Convert a list of lines and heading Sentence objects into Sentence objects.

        Args:
            lines: List of strings or Sentence objects.

        Returns:
            List of Sentence objects.
        """
        text = ""
        sentence_objects = []

        for line in lines:
            if isinstance(line, Sentence):
                if text:
                    sentences = self.tokenize_sentences(text)
                    sentence_objects += map(self._to_sentence, sentences)

                sentence_objects.append(line)
                text = ""
            else:
                text += " " + line

        text = text.strip()
        if text:
            sentences = self.tokenize_sentences(text)
            sentence_objects += map(self._to_sentence, sentences)

        return sentence_objects

    def _to_sentence(self, text: str) -> Sentence:
        """Create a Sentence from text.

        Args:
            text: Sentence text.

        Returns:
            Sentence instance.
        """
        assert text.strip()
        return Sentence(text, self._tokenizer)
