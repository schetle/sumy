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
        
        for element in self._iter_elements_with_tags(tree):
            if "a" in element.tag or "strike" in element.tag or "s" in element.tag:
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

    def _iter_elements_with_tags(self, tree):
        """Iterate over elements that might have stigma tags."""
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
                if element.text and element.text.strip():
                    text = element.text.strip()
                    sentences = self.tokenize_sentences(text)
                    for s in sentences:
                        current_sentences.append(Sentence(s, self._tokenizer))
                # Process any nested elements within the paragraph
                for child in element:
                    if child.text and child.text.strip():
                        text = child.text.strip()
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
