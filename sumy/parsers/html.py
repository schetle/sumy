from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING
from urllib.request import urlopen
import lxml.html
from readability import Document

from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from .parser import DocumentParser

if TYPE_CHECKING:
    from ..nlp.tokenizers import Tokenizer


class HtmlParser(DocumentParser):
    """Parser of text from HTML format into DOM."""

    SIGNIFICANT_TAGS = ("h1", "h2", "h3", "b", "strong", "big", "dfn", "em")
    STIGMA_TAGS = ("a", "strike", "s")
    HEADING_TAGS = ("h1", "h2", "h3")

    @classmethod
    def from_string(cls, string: str | bytes, url: str | None, tokenizer: Tokenizer) -> HtmlParser:
        return cls(string, tokenizer, url)

    @classmethod
    def from_file(cls, file_path: str, url: str | None, tokenizer: Tokenizer) -> HtmlParser:
        with open(file_path, "rb") as f:
            return cls(f.read(), tokenizer, url)

    @classmethod
    def from_url(cls, url: str, tokenizer: Tokenizer) -> HtmlParser:
        response = urlopen(url)
        data = response.read()
        response.close()
        return cls(data, tokenizer, url)

    def __init__(self, html_content: str | bytes, tokenizer: Tokenizer, url: str | None = None) -> None:
        super().__init__(tokenizer)
        if isinstance(html_content, bytes):
            html_content = html_content.decode("utf-8", errors="replace")
        doc = Document(html_content)
        summary_html = doc.summary(html_partial=True)
        self._root = lxml.html.fromstring(summary_html)

    @cached_property
    def significant_words(self) -> tuple[str, ...]:
        words = []
        for element in self._root.iter():
            if element.tag in self.SIGNIFICANT_TAGS and element.text_content().strip():
                words.extend(self.tokenize_words(element.text_content()))
        return tuple(words) if words else self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self) -> tuple[str, ...]:
        words = []
        for element in self._root.iter():
            if element.tag in self.STIGMA_TAGS and element.text_content().strip():
                words.extend(self.tokenize_words(element.text_content()))
        return tuple(words) if words else self.STIGMA_WORDS

    @cached_property
    def document(self) -> ObjectDocumentModel:
        paragraphs = []
        for element in self._root.iter():
            if element.tag in ("p", "h1", "h2", "h3"):
                text = element.text_content().strip()
                if not text:
                    continue
                sentences = []
                is_heading = element.tag in self.HEADING_TAGS
                if is_heading:
                    sentences.append(Sentence(text, self._tokenizer, is_heading=True))
                else:
                    for s in self.tokenize_sentences(text):
                        sentences.append(Sentence(s, self._tokenizer))
                if sentences:
                    paragraphs.append(Paragraph(sentences))
        return ObjectDocumentModel(paragraphs)
