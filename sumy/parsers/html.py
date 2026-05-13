"""Parser for HTML documents using breadability for article extraction."""

from urllib import request as urllib
from breadability.readable import Article
from functools import cached_property
from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from .parser import DocumentParser


class HtmlParser(DocumentParser):
    """Parser of text from HTML format into DOM.

    Uses breadability to extract the main article content from HTML pages.
    """

    SIGNIFICANT_TAGS = (
        "h1", "h2", "h3",
        "b", "strong",
        "big",
        "dfn",
        "em",
    )

    @classmethod
    def from_string(cls, string: str, url: str, tokenizer) -> "HtmlParser":
        """Create a parser from an HTML string.

        Args:
            string: HTML content.
            url: URL of the original page.
            tokenizer: Tokenizer instance.

        Returns:
            HtmlParser instance.
        """
        return cls(string, tokenizer, url)

    @classmethod
    def from_file(cls, file_path: str, url: str, tokenizer) -> "HtmlParser":
        """Create a parser from an HTML file.

        Args:
            file_path: Path to the HTML file.
            url: URL of the original page.
            tokenizer: Tokenizer instance.

        Returns:
            HtmlParser instance.
        """
        with open(file_path, "rb") as file:
            return cls(file.read(), tokenizer, url)

    @classmethod
    def from_url(cls, url: str, tokenizer) -> "HtmlParser":
        """Create a parser by fetching HTML from a URL.

        Args:
            url: URL to fetch HTML from.
            tokenizer: Tokenizer instance.

        Returns:
            HtmlParser instance.
        """
        response = urllib.urlopen(url)
        data = response.read()
        response.close()

        return cls(data, tokenizer, url)

    def __init__(self, html_content, tokenizer, url=None):
        """Initialize an HtmlParser.

        Args:
            html_content: HTML content as string or bytes.
            tokenizer: Tokenizer instance.
            url: Optional URL of the original page.
        """
        super().__init__(tokenizer)
        self._article = Article(html_content, url)

    @cached_property
    def significant_words(self) -> tuple:
        """Return words from significant HTML tags (headings, bold, emphasis).

        Returns:
            Tuple of significant words, or default significant words if none found.
        """
        words = []
        for paragraph in self._article.main_text:
            for text, annotations in paragraph:
                if self._contains_any(annotations, *self.SIGNIFICANT_TAGS):
                    words.extend(self.tokenize_words(text))

        if words:
            return tuple(words)
        else:
            return self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self) -> tuple:
        """Return words from stigma HTML tags (links, strikethrough).

        Returns:
            Tuple of stigma words, or default stigma words if none found.
        """
        words = []
        for paragraph in self._article.main_text:
            for text, annotations in paragraph:
                if self._contains_any(annotations, "a", "strike", "s"):
                    words.extend(self.tokenize_words(text))

        if words:
            return tuple(words)
        else:
            return self.STIGMA_WORDS

    def _contains_any(self, sequence, *args) -> bool:
        """Check if sequence contains any of the given items.

        Args:
            sequence: Sequence to check (may be None).
            *args: Items to look for.

        Returns:
            True if any item is found in the sequence.
        """
        if sequence is None:
            return False

        for item in args:
            if item in sequence:
                return True

        return False

    @cached_property
    def document(self) -> ObjectDocumentModel:
        """Parse the HTML into an ObjectDocumentModel.

        Returns:
            ObjectDocumentModel with paragraphs, sentences, and words.
        """
        annotated_text = self._article.main_text

        paragraphs = []
        for paragraph in annotated_text:
            sentences = []

            current_text = ""
            for text, annotations in paragraph:
                if annotations and ("h1" in annotations or "h2" in annotations or "h3" in annotations):
                    sentences.append(Sentence(text, self._tokenizer, is_heading=True))
                # skip <pre> nodes
                elif not (annotations and "pre" in annotations):
                    current_text += " " + text

            new_sentences = self.tokenize_sentences(current_text)
            sentences.extend(Sentence(s, self._tokenizer) for s in new_sentences)
            paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)
