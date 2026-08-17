import unittest

from sumy.parsers.plaintext import PlaintextParser
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from .utils import expand_resource_path


class TestParser(unittest.TestCase):
    def test_parse_plaintext(self):
        parser = PlaintextParser.from_string("""
            Ako sa máš? Ja dobre! A ty? No
            mohlo to byť aj lepšie!!! Ale pohodička.


            TOTO JE AKOŽE NADPIS
            A toto je text pod ním, ktorý je textový.
            A tak ďalej...
        """, Tokenizer("czech"))

        document = parser.document

        self.assertEqual(len(document.paragraphs), 2)

        self.assertEqual(len(document.paragraphs[0].headings), 0)
        self.assertEqual(len(document.paragraphs[0].sentences), 5)

        self.assertEqual(len(document.paragraphs[1].headings), 1)
        self.assertEqual(len(document.paragraphs[1].sentences), 2)

    def test_parse_plaintext_long(self):
        parser = PlaintextParser.from_string("""
            Ako sa máš? Ja dobre! A ty? No
            mohlo to byť aj lepšie!!! Ale pohodička.

            TOTO JE AKOŽE NADPIS
            A toto je text pod ním, ktorý je textový.
            A tak ďalej...

            VEĽKOLEPÉ PREKVAPENIE
            Tretí odstavec v tomto texte je úplne o ničom. Ale má
            vety a to je hlavné. Takže sa majte na pozore ;-)

            A tak ďalej...


            A tak este dalej!
        """, Tokenizer("czech"))

        document = parser.document

        self.assertEqual(len(document.paragraphs), 5)

        self.assertEqual(len(document.paragraphs[0].headings), 0)
        self.assertEqual(len(document.paragraphs[0].sentences), 5)

        self.assertEqual(len(document.paragraphs[1].headings), 1)
        self.assertEqual(len(document.paragraphs[1].sentences), 2)

        self.assertEqual(len(document.paragraphs[2].headings), 1)
        self.assertEqual(len(document.paragraphs[2].sentences), 3)

        self.assertEqual(len(document.paragraphs[3].headings), 0)
        self.assertEqual(len(document.paragraphs[3].sentences), 1)

        self.assertEqual(len(document.paragraphs[4].headings), 0)
        self.assertEqual(len(document.paragraphs[4].sentences), 1)


class TestHtmlParser(unittest.TestCase):
    def test_html_document_has_paragraphs(self):
        path = expand_resource_path("snippets/paragraphs.html")
        url = "http://www.snippet.org/paragraphs.html"
        parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

        document = parser.document

        self.assertGreater(len(document.paragraphs), 0)
        self.assertGreater(len(document.sentences), 0)

    def test_html_document_extracts_headings(self):
        path = expand_resource_path("snippets/paragraphs.html")
        url = "http://www.snippet.org/paragraphs.html"
        parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

        document = parser.document

        heading_texts = [str(s) for s in document.headings]
        self.assertIn("Toto je nadpis prvej úrovne", heading_texts)

    def test_html_document_extracts_sentences(self):
        path = expand_resource_path("snippets/paragraphs.html")
        url = "http://www.snippet.org/paragraphs.html"
        parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

        document = parser.document

        sentence_texts = [str(s) for s in document.sentences]
        self.assertTrue(
            any("Toto je prvý odstavec" in t for t in sentence_texts)
        )

    def test_html_significant_words_from_headings(self):
        path = expand_resource_path("snippets/paragraphs.html")
        url = "http://www.snippet.org/paragraphs.html"
        parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

        sig = parser.significant_words
        self.assertIsInstance(sig, tuple)
        self.assertGreater(len(sig), 0)

    def test_html_stigma_words_fallback(self):
        path = expand_resource_path("snippets/paragraphs.html")
        url = "http://www.snippet.org/paragraphs.html"
        parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

        stigma = parser.stigma_words
        self.assertIsInstance(stigma, tuple)
        self.assertGreater(len(stigma), 0)
