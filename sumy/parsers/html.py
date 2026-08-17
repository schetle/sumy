import urllib.request
from functools import cached_property

import trafilatura
from lxml import etree

from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from .parser import DocumentParser


class HtmlParser(DocumentParser):
    """Parser of text from HTML format into DOM."""

    @classmethod
    def from_string(cls, string, url, tokenizer):
        return cls(string, tokenizer, url)

    @classmethod
    def from_file(cls, file_path, url, tokenizer):
        with open(file_path, "rb") as f:
            return cls(f.read(), tokenizer, url)

    @classmethod
    def from_url(cls, url, tokenizer):
        with urllib.request.urlopen(url) as response:
            data = response.read()
        return cls(data, tokenizer, url)

    def __init__(self, html_content, tokenizer, url=None):
        super().__init__(tokenizer)
        if isinstance(html_content, bytes):
            html_content = html_content.decode("utf-8", errors="replace")
        self._html_content = html_content
        self._url = url

    @cached_property
    def _extract_xml_root(self):
        xml_str = trafilatura.extract(
            self._html_content,
            output_format="xml",
            include_formatting=True,
            include_links=False,
            url=self._url,
        )
        if not xml_str:
            return None
        try:
            return etree.fromstring(xml_str.encode("utf-8"))
        except etree.XMLSyntaxError:
            return None

    @staticmethod
    def _tag_name(element):
        tag = element.tag
        return tag.split("}")[-1] if "}" in tag else tag

    @cached_property
    def significant_words(self):
        root = self._extract_xml_root
        if root is None:
            return self.SIGNIFICANT_WORDS

        words = []
        for element in root.iter():
            if self._tag_name(element) == "head":
                text = (element.text or "").strip()
                if text:
                    words.extend(self.tokenize_words(text))

        return tuple(words) if words else self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self):
        return self.STIGMA_WORDS

    @cached_property
    def document(self):
        root = self._extract_xml_root
        if root is None:
            return ObjectDocumentModel([])

        paragraphs = []
        for element in root.iter():
            tag = self._tag_name(element)
            text = (element.text or "").strip()
            if not text:
                continue

            if tag == "head":
                paragraphs.append(Paragraph([Sentence(text, self._tokenizer, is_heading=True)]))
            elif tag == "p":
                sentences = [
                    Sentence(s, self._tokenizer)
                    for s in self.tokenize_sentences(text)
                ]
                if sentences:
                    paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)
