from readability import Document
from lxml import html as lxml_html
from urllib import request as urllib
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
        response = urllib.urlopen(url)
        data = response.read()
        response.close()
        return cls(data, tokenizer, url)

    def __init__(self, html_content, tokenizer, url=None):
        super().__init__(tokenizer)
        if isinstance(html_content, bytes):
            html_content = html_content.decode("utf-8", errors="replace")
        doc = Document(html_content, url=url)
        self._summary_html = doc.summary(html_partial=True)
        # Keep the original html for fallback if summary is too short
        self._original_html = html_content

    @cached_property
    def significant_words(self):
        try:
            tree = lxml_html.fromstring(self._summary_html or self._original_html)
        except Exception:
            return self.SIGNIFICANT_WORDS
        words = []
        for tag in self.SIGNIFICANT_TAGS:
            for el in tree.xpath(f"//{tag}"):
                text = (el.text_content() or "").strip()
                if text:
                    words.extend(self.tokenize_words(text))
        return tuple(words) if words else self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self):
        try:
            tree = lxml_html.fromstring(self._summary_html or self._original_html)
        except Exception:
            return self.STIGMA_WORDS
        words = []
        for tag in ("a", "strike", "s"):
            for el in tree.xpath(f"//{tag}"):
                text = (el.text_content() or "").strip()
                if text:
                    words.extend(self.tokenize_words(text))
        return tuple(words) if words else self.STIGMA_WORDS

    @cached_property
    def document(self):
        try:
            tree = lxml_html.fromstring(self._original_html)
        except Exception:
            return ObjectDocumentModel([])

        heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
        block_tags = {"p", "li", "blockquote", "td", "th"}
        skip_tags = {"script", "style", "pre"}

        paragraphs = []
        # Collect all relevant elements in document order, then group:
        # a heading and the block elements that immediately follow it
        # (until the next heading) form one paragraph.
        # Block elements that come before any heading form their own paragraphs.

        # Gather sequence of (kind, element) tuples
        elements = []
        for el in tree.iter():
            tag = el.tag if isinstance(el.tag, str) else ""
            tag = tag.lower()

            if tag in skip_tags:
                continue

            if tag in heading_tags:
                elements.append(("heading", el))
            elif tag in block_tags:
                # Only process leaf-ish block elements (not containers of other blocks)
                child_tags = {c.tag.lower() for c in el if isinstance(c.tag, str)}
                if child_tags & (block_tags | heading_tags):
                    continue  # skip container elements
                elements.append(("block", el))

        # Group: each block element forms its own paragraph; if the immediately
        # preceding element was a heading, that heading is prepended into the
        # same paragraph.  Consecutive headings without an intervening block are
        # each emitted as a standalone paragraph.
        groups = []
        pending_headings = []

        for kind, el in elements:
            if kind == "heading":
                pending_headings.append(("heading", el))
            else:
                # block element — attach any pending headings to this paragraph
                group = pending_headings + [("block", el)]
                pending_headings = []
                groups.append(group)

        # Flush any trailing headings (no following block) as individual paragraphs
        for heading in pending_headings:
            groups.append([heading])

        # Build paragraphs from groups
        for group in groups:
            sentences = []
            block_texts = []

            for kind, el in group:
                text = (el.text_content() or "").strip()
                if not text:
                    continue
                if kind == "heading":
                    sentences.append(Sentence(text, self._tokenizer, is_heading=True))
                else:
                    block_texts.append(text)

            # Tokenize all collected block text
            combined = " ".join(block_texts)
            new_sentences = self.tokenize_sentences(combined)
            sentences.extend(Sentence(s, self._tokenizer) for s in new_sentences)

            if sentences:
                paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)
