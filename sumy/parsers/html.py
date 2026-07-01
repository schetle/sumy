from collections.abc import Generator
from typing import Any, Self

import lxml.html
from readability import Document as ReadabilityDocument
from urllib import request as urllib
from ..utils import cached_property
from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from ..nlp.tokenizers import Tokenizer
from .parser import DocumentParser


# Block-level elements that start a new paragraph context
_BLOCK_TAGS: frozenset[str] = frozenset({
    "p", "div", "blockquote", "ul", "ol", "dl", "table",
    "article", "section", "aside", "header", "footer",
    "li", "dd", "dt", "tr", "td", "th", "figure", "figcaption",
    "details", "summary", "main", "nav", "form", "fieldset",
})

_HEADING_TAGS: frozenset[str] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_SENTENCE_HEADING_TAGS: frozenset[str] = frozenset({"h1", "h2", "h3"})

_SKIP_TAGS: frozenset[str] = frozenset({"pre", "script", "style"})


def _has_block_children(element: Any) -> bool:
    """Return True if *element* has any direct child that is a block or heading."""
    for child in element:
        if isinstance(child.tag, str) and (child.tag in _BLOCK_TAGS or child.tag in _HEADING_TAGS):
            return True
    return False


def _iter_leaf_blocks(element: Any) -> Generator[Any, None, None]:
    """Yield leaf block-level elements -- blocks with no block children.

    This avoids double-counting content inside nested ``<div>`` wrappers:
    a ``<div>`` that contains ``<p>`` children is skipped, but each ``<p>``
    is yielded. A ``<div>`` that contains only inline content is yielded
    directly.
    """
    tag = element.tag
    if not isinstance(tag, str):
        return

    is_block = tag in _BLOCK_TAGS or tag in _HEADING_TAGS
    if is_block and not _has_block_children(element):
        yield element
    else:
        for child in element:
            yield from _iter_leaf_blocks(child)


def _iter_text_with_tags(element: Any) -> Generator[tuple[str, frozenset[str]], None, None]:
    """Yield ``(text, tag_set)`` pairs for every text node under *element*.

    *tag_set* is a frozenset of the tag names of all ancestor elements that
    wrap the text, relative to (and including) *element* itself.
    """
    # Direct text of the element itself
    if element.text and element.text.strip():
        yield element.text.strip(), frozenset({element.tag})

    for child in element:
        if child.tag in _SKIP_TAGS:
            # Yield the tail (text after this skipped element) if any
            if child.tail and child.tail.strip():
                yield child.tail.strip(), frozenset({element.tag})
            continue

        # Recurse into child, adding child.tag to the ancestor set
        for text, tags in _iter_text_with_tags(child):
            yield text, tags | {element.tag}

        # Tail text belongs to the parent element scope
        if child.tail and child.tail.strip():
            yield child.tail.strip(), frozenset({element.tag})


class HtmlParser(DocumentParser):
    """Parser of text from HTML format into DOM."""

    SIGNIFICANT_TAGS: tuple[str, ...] = (
        "h1", "h2", "h3",
        "b", "strong",
        "big",
        "dfn",
        "em",
    )

    @classmethod
    def from_string(cls, string: str | bytes, url: str, tokenizer: Tokenizer) -> Self:
        return cls(string, tokenizer, url)

    @classmethod
    def from_file(cls, file_path: str, url: str, tokenizer: Tokenizer) -> Self:
        with open(file_path, "rb") as file:
            return cls(file.read(), tokenizer, url)

    @classmethod
    def from_url(cls, url: str, tokenizer: Tokenizer) -> Self:
        response = urllib.urlopen(url)
        data = response.read()
        response.close()

        return cls(data, tokenizer, url)

    def __init__(self, html_content: str | bytes, tokenizer: Tokenizer, url: str | None = None) -> None:
        super().__init__(tokenizer)
        if isinstance(html_content, bytes):
            html_content = html_content.decode("utf-8", errors="replace")
        self._readable_html: str = ReadabilityDocument(html_content, url=url).summary()

    def _parse_tree(self) -> Any:
        """Return the lxml tree for the readable HTML."""
        return lxml.html.fromstring(self._readable_html)

    # ------------------------------------------------------------------
    # Annotated-text helpers
    # ------------------------------------------------------------------

    @cached_property
    def _build_annotated_paragraphs(self) -> list[list[tuple[str, frozenset[str]]]]:
        """Walk the readable HTML and return a list of *annotated paragraphs*.

        Each annotated paragraph is a list of ``(text, tags_frozenset)`` tuples
        that mirrors what ``breadability``'s ``article.main_text`` used to
        provide.

        Heading elements (``h1``-``h3``) that immediately precede a body
        block are grouped into the same paragraph so the Edmundson
        summariser can use ``paragraph.headings``.
        """
        root = self._parse_tree()
        paragraphs: list[list[tuple[str, frozenset[str]]]] = []
        current: list[tuple[str, frozenset[str]]] = []  # accumulator for the current paragraph

        for element in _iter_leaf_blocks(root):
            tag = element.tag

            if tag in _HEADING_TAGS:
                # Headings start a new paragraph group (heading + following body)
                if current:
                    paragraphs.append(current)
                current = list(_iter_text_with_tags(element))
            elif tag in _BLOCK_TAGS:
                pieces = list(_iter_text_with_tags(element))
                if not pieces:
                    continue
                if current:
                    # There is a pending heading -- attach this block to it
                    current.extend(pieces)
                    paragraphs.append(current)
                    current = []
                else:
                    paragraphs.append(pieces)

        # Flush any trailing heading-only paragraph
        if current:
            paragraphs.append(current)

        return paragraphs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _contains_any(self, sequence: frozenset[str] | None, *args: str) -> bool:
        if sequence is None:
            return False

        for item in args:
            if item in sequence:
                return True

        return False

    @cached_property
    def significant_words(self) -> tuple[str, ...]:
        words: list[str] = []
        for paragraph in self._build_annotated_paragraphs:
            for text, tags in paragraph:
                if self._contains_any(tags, *self.SIGNIFICANT_TAGS):
                    words.extend(self.tokenize_words(text))

        if words:
            return tuple(words)
        else:
            return self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self) -> tuple[str, ...]:
        words: list[str] = []
        for paragraph in self._build_annotated_paragraphs:
            for text, tags in paragraph:
                if self._contains_any(tags, "a", "strike", "s"):
                    words.extend(self.tokenize_words(text))

        if words:
            return tuple(words)
        else:
            return self.STIGMA_WORDS

    @cached_property
    def document(self) -> ObjectDocumentModel:
        annotated_paragraphs = self._build_annotated_paragraphs

        paragraphs: list[Paragraph] = []
        for paragraph in annotated_paragraphs:
            sentences: list[Sentence] = []

            current_text = ""
            for text, tags in paragraph:
                if tags and tags & _SENTENCE_HEADING_TAGS:
                    sentences.append(Sentence(text, self._tokenizer, is_heading=True))
                # skip <pre> nodes
                elif not (tags and "pre" in tags):
                    current_text += " " + text

            new_sentences = self.tokenize_sentences(current_text)
            sentences.extend(Sentence(s, self._tokenizer) for s in new_sentences)
            paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)
