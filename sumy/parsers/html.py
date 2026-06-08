# -*- coding: utf8 -*-

import urllib.request

from readability import Document
from lxml import etree

from ..utils import cached_property
from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from .parser import DocumentParser


class HtmlParser(DocumentParser):
    """Parser of text from HTML format into DOM."""

    SIGNIFICANT_TAGS = (
        "h1", "h2", "h3",
        "b", "strong",
        "big",
        "dfn",
        "em",
    )

    @classmethod
    def from_string(cls, string, url, tokenizer):
        return cls(string, tokenizer, url)

    @classmethod
    def from_file(cls, file_path, url, tokenizer):
        with open(file_path, "rb") as file:
            return cls(file.read(), tokenizer, url)

    @classmethod
    def from_url(cls, url, tokenizer):
        response = urllib.request.urlopen(url)
        data = response.read()
        response.close()
        return cls(data, tokenizer, url)

    def __init__(self, html_content, tokenizer, url=None):
        super().__init__(tokenizer)
        if isinstance(html_content, bytes):
            html_content = html_content.decode('utf-8', errors='replace')
        doc = Document(html_content)
        self._summary_html = doc.summary()
        self._url = url

    @cached_property
    def _parse_summary(self):
        """Parse the readability summary HTML into an lxml element tree."""
        return etree.fromstring(self._summary_html, etree.HTMLParser())

    @cached_property
    def significant_words(self):
        words = []
        root = self._parse_summary
        for tag in self.SIGNIFICANT_TAGS:
            for element in root.iter(tag):
                if element.text:
                    words.extend(self.tokenize_words(element.text))
                # also check tail text in child elements
                for child in element:
                    if child.tail:
                        words.extend(self.tokenize_words(child.tail))
        if words:
            return tuple(words)
        else:
            return self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self):
        words = []
        root = self._parse_summary
        for tag in ("a", "strike", "s"):
            for element in root.iter(tag):
                if element.text:
                    words.extend(self.tokenize_words(element.text))
        if words:
            return tuple(words)
        else:
            return self.STIGMA_WORDS

    @cached_property
    def document(self):
        root = self._parse_summary
        paragraphs = []

        # readability-lxml wraps content in a div inside body; use iter to find
        # all h1/h2/h3/p elements in document order regardless of nesting depth
        for element in root.iter():
            tag = element.tag.lower() if isinstance(element.tag, str) else ''
            sentences = []

            if tag in ('h1', 'h2', 'h3'):
                text = etree.tostring(element, method='text', encoding='unicode').strip()
                if text:
                    sentences.append(Sentence(text, self._tokenizer, is_heading=True))
            elif tag == 'p':
                # Get all text content from the paragraph
                text = etree.tostring(element, method='text', encoding='unicode').strip()
                if text:
                    new_sentences = self.tokenize_sentences(text)
                    sentences.extend(Sentence(s, self._tokenizer) for s in new_sentences)

            if sentences:
                paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)
