import unittest

from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence
from sumy.nlp.tokenizers import Tokenizer

UNICODE_STRING = "ľščťžáýíééäúňô €đ€Ł¤"
BYTES_STRING = UNICODE_STRING.encode("utf8")


class TestSentenceStringConversion(unittest.TestCase):
    """Native ``__str__``/``__repr__`` of DOM classes replaced the former ``unicode_compatible`` helper."""

    def setUp(self) -> None:
        self.sentence = Sentence(UNICODE_STRING, Tokenizer("czech"))

    def test_str(self) -> None:
        self.assertEqual(str(self.sentence), UNICODE_STRING)

    def test_repr(self) -> None:
        self.assertEqual(repr(self.sentence), f"<Sentence: {UNICODE_STRING}>")

    def test_heading_repr(self) -> None:
        heading = Sentence(UNICODE_STRING, Tokenizer("czech"), is_heading=True)
        self.assertEqual(repr(heading), f"<Heading: {UNICODE_STRING}>")

    def test_bytes_text_is_decoded(self) -> None:
        sentence = Sentence(BYTES_STRING, Tokenizer("czech"))
        self.assertEqual(str(sentence), UNICODE_STRING)

    def test_paragraph_str(self) -> None:
        paragraph = Paragraph([self.sentence])
        self.assertEqual(str(paragraph), "<Paragraph with 0 headings & 1 sentences>")
        self.assertEqual(repr(paragraph), str(paragraph))

    def test_document_str(self) -> None:
        document = ObjectDocumentModel([Paragraph([self.sentence])])
        self.assertEqual(str(document), "<DOM with 1 paragraphs>")
        self.assertEqual(repr(document), str(document))
