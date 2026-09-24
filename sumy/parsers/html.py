from urllib import request as urllib

from lxml import html as lxml_html
from readability import Document

from .parser import DocumentParser
from ..models.dom import Sentence, Paragraph, ObjectDocumentModel


class HtmlParser(DocumentParser):
    """Parser of text from HTML format into DOM."""

    SIGNIFICANT_TAGS = (
        "h1", "h2", "h3",
        "b", "strong",
        "big",
        "dfn",
        "em",
    )

    STIGMA_TAGS = ("a", "strike", "s")

    @classmethod
    def from_string(cls, string, url, tokenizer):
        return cls(string, tokenizer, url)

    @classmethod
    def from_file(cls, file_path, url, tokenizer):
        with open(file_path, "rb") as file:
            return cls(file.read(), tokenizer, url)

    @classmethod
    def from_url(cls, url, tokenizer):
        response = urllib.urlopen(url)
        data = response.read()
        response.close()

        return cls(data, tokenizer, url)

    def __init__(self, html_content, tokenizer, url=None):
        super(HtmlParser, self).__init__(tokenizer)
        self._html_content = html_content
        if isinstance(html_content, bytes):
            self._html_content = html_content.decode("utf8")
        self._url = url
        self._article = Document(self._html_content)

    @property
    def significant_words(self):
        summary_html = self._article.summary()
        tree = lxml_html.fromstring(summary_html)
        words = []
        
        for element in self._iter_text_elements(tree):
            if self._is_significant_element(element):
                words.extend(self.tokenize_words(element.text or ""))
        
        if words:
            return tuple(words)
        else:
            return self.SIGNIFICANT_WORDS

    @property
    def stigma_words(self):
        summary_html = self._article.summary()
        tree = lxml_html.fromstring(summary_html)
        words = []
        
        for element in self._iter_text_elements(tree):
            if element.tag in self.STIGMA_TAGS:
                words.extend(self.tokenize_words(element.text or ""))
        
        if words:
            return tuple(words)
        else:
            return self.STIGMA_WORDS

    def _iter_text_elements(self, tree):
        """Iterate over elements that contain significant text."""
        for element in tree.iter():
            if element.text and element.text.strip():
                yield element

    def _is_significant_element(self, element):
        """Check if element has significant tags."""
        # Check the element's own tag
        if element.tag in self.SIGNIFICANT_TAGS:
            return True
        # Check parent tags for heading context
        parent = element.getparent()
        if parent is not None:
            if parent.tag in ("h1", "h2", "h3"):
                return True
        return False

    @property
    def document(self):
        summary_html = self._article.summary()
        tree = lxml_html.fromstring(summary_html)
        
        paragraphs = []
        current_sentences = []
        current_headings = []
        
        for element in tree.iter():
            if element.tag in ("h1", "h2", "h3"):
                if element.text and element.text.strip():
                    heading = Sentence(element.text.strip(), self._tokenizer, is_heading=True)
                    current_headings.append(heading)
            elif element.tag == "p":
                # Collect text from paragraph and its children
                text_parts = []
                if element.text and element.text.strip():
                    text_parts.append(element.text.strip())
                for child in element:
                    if child.text and child.text.strip():
                        text_parts.append(child.text.strip())
                    if child.tail and child.tail.strip():
                        text_parts.append(child.tail.strip())
                
                if text_parts:
                    text = " ".join(text_parts)
                    sentences = self.tokenize_sentences(text)
                    for s in sentences:
                        current_sentences.append(Sentence(s, self._tokenizer))
                
                # Create paragraph with accumulated sentences
                if current_sentences or current_headings:
                    all_sentences = current_headings + current_sentences
                    paragraphs.append(Paragraph(all_sentences))
                    current_sentences = []
                    current_headings = []
        
        # Handle any remaining content
        if current_sentences or current_headings:
            all_sentences = current_headings + current_sentences
            paragraphs.append(Paragraph(all_sentences))
        
        return ObjectDocumentModel(paragraphs)
