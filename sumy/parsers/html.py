"""Parser for HTML documents using readability-lxml for article extraction."""

from urllib import request as urllib
from functools import cached_property

from readability import Document
import lxml.html

from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from .parser import DocumentParser


class HtmlParser(DocumentParser):
    """Parser of text from HTML format into DOM.

    Uses readability-lxml to extract the main article content from HTML pages,
    then walks the cleaned HTML tree with lxml to reconstruct paragraph
    structure with tag annotations for heading/emphasis detection.
    """

    SIGNIFICANT_TAGS = (
        "h1", "h2", "h3",
        "b", "strong",
        "big",
        "dfn",
        "em",
    )

    HEADING_TAGS = frozenset({"h1", "h2", "h3"})
    STIGMA_TAGS = frozenset({"a", "strike", "s"})
    SKIP_TAGS = frozenset({"pre"})
    PARAGRAPH_TAGS = frozenset({"p", "div", "article", "section", "blockquote", "li"})

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
        if isinstance(html_content, bytes):
            html_content = html_content.decode("utf8", errors="replace")
        self._document = Document(html_content, url=url)

    def _get_annotated_text(self):
        """Extract annotated text from the readability-lxml cleaned HTML.

        Walks the lxml element tree and produces a list of paragraphs,
        where each paragraph is a list of (text, annotations) tuples.
        The annotations indicate which HTML tags wrap each text fragment.

        Returns:
            List of paragraphs, each a list of (text, tag_set_or_None) tuples.
        """
        summary_html = self._document.summary()
        try:
            root = lxml.html.fromstring(summary_html)
        except Exception:
            return []

        paragraphs = []
        current_paragraph = []

        def _get_tag_context(element):
            """Get the set of ancestor tag names for an element.

            Args:
                element: lxml element.

            Returns:
                Set of tag name strings, or None if empty.
            """
            tags = set()
            el = element
            while el is not None:
                if hasattr(el, "tag") and isinstance(el.tag, str):
                    tags.add(el.tag.lower())
                el = el.getparent()
            tags.discard("html")
            tags.discard("body")
            tags.discard("div")
            return tags if tags else None

        def _flush_paragraph():
            """Flush the current paragraph if it has content."""
            if current_paragraph:
                paragraphs.append(list(current_paragraph))
                current_paragraph.clear()

        def _walk(element):
            """Recursively walk the element tree extracting text with annotations.

            Args:
                element: lxml element to process.
            """
            tag = element.tag.lower() if isinstance(element.tag, str) else ""

            # Check if this is a paragraph-level element
            is_paragraph_break = tag in self.PARAGRAPH_TAGS or tag in self.HEADING_TAGS

            if is_paragraph_break:
                _flush_paragraph()

            # Process element's own text
            if element.text and element.text.strip():
                annotations = _get_tag_context(element)
                current_paragraph.append((element.text.strip(), annotations))

            # Process children
            for child in element:
                _walk(child)
                # Process tail text (text after child element but before next sibling)
                if child.tail and child.tail.strip():
                    annotations = _get_tag_context(element)
                    current_paragraph.append((child.tail.strip(), annotations))

            if is_paragraph_break:
                _flush_paragraph()

        _walk(root)
        _flush_paragraph()

        return paragraphs

    @cached_property
    def significant_words(self) -> tuple:
        """Return words from significant HTML tags (headings, bold, emphasis).

        Returns:
            Tuple of significant words, or default significant words if none found.
        """
        words = []
        for paragraph in self._get_annotated_text():
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
        for paragraph in self._get_annotated_text():
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
        annotated_text = self._get_annotated_text()

        paragraphs = []
        for paragraph in annotated_text:
            sentences = []

            current_text = ""
            for text, annotations in paragraph:
                if annotations and self.HEADING_TAGS.intersection(annotations):
                    sentences.append(Sentence(text, self._tokenizer, is_heading=True))
                # skip <pre> nodes
                elif not (annotations and self.SKIP_TAGS.intersection(annotations)):
                    current_text += " " + text

            new_sentences = self.tokenize_sentences(current_text)
            sentences.extend(Sentence(s, self._tokenizer) for s in new_sentences)
            paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)
