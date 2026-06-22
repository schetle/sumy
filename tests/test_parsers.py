import unittest
from unittest.mock import patch, MagicMock

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
    def test_annotated_text(self):
        path = expand_resource_path("snippets/paragraphs.html")
        url = "http://www.snippet.org/paragraphs.html"
        parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

        document = parser.document

        self.assertEqual(len(document.paragraphs), 2)

        self.assertEqual(len(document.paragraphs[0].headings), 1)
        self.assertEqual(len(document.paragraphs[0].sentences), 1)

        self.assertEqual(str(document.paragraphs[0].headings[0]),
            "Toto je nadpis prvej úrovne")
        self.assertEqual(str(document.paragraphs[0].sentences[0]),
            "Toto je prvý odstavec a to je fajn.")

        self.assertEqual(len(document.paragraphs[1].headings), 0)
        self.assertEqual(len(document.paragraphs[1].sentences), 2)

        self.assertEqual(str(document.paragraphs[1].sentences[0]),
            "Tento text je tu aby vyplnil prázdne miesto v srdci súboru.")
        self.assertEqual(str(document.paragraphs[1].sentences[1]),
            "Aj súbory majú predsa city.")

    def test_from_string(self):
        html = "<html><body><p>First sentence. Second sentence.</p></body></html>"
        parser = HtmlParser.from_string(html, "http://example.com", Tokenizer("english"))
        document = parser.document
        self.assertGreater(len(document.paragraphs), 0)

    def test_from_url(self):
        html = b"<html><body><p>Test sentence. Another sentence here.</p></body></html>"
        mock_response = MagicMock()
        mock_response.read.return_value = html
        mock_response.close.return_value = None
        with patch("sumy.parsers.html.urllib.urlopen", return_value=mock_response):
            parser = HtmlParser.from_url("http://example.com", Tokenizer("english"))
        document = parser.document
        self.assertGreater(len(document.paragraphs), 0)

    def test_significant_words_with_marked_text(self):
        html = "<html><body><p>Normal text. <strong>Important word here.</strong></p></body></html>"
        parser = HtmlParser.from_string(html, None, Tokenizer("english"))
        words = parser.significant_words
        self.assertIsInstance(words, tuple)

    def test_significant_words_fallback_on_empty(self):
        html = "<html><body><p>No headings or bold text here.</p></body></html>"
        parser = HtmlParser.from_string(html, None, Tokenizer("english"))
        words = parser.significant_words
        self.assertIsInstance(words, tuple)

    def test_significant_words_fallback_on_bad_html(self):
        html = "<html><body><p>Some text here.</p></body></html>"
        parser = HtmlParser.from_string(html, None, Tokenizer("english"))
        with patch("sumy.parsers.html.lxml_html.fromstring", side_effect=Exception("parse error")):
            words = parser._words_from_tags(HtmlParser.SIGNIFICANT_TAGS, HtmlParser.SIGNIFICANT_WORDS)
        self.assertEqual(words, HtmlParser.SIGNIFICANT_WORDS)

    def test_stigma_words_with_links(self):
        html = "<html><body><p>Normal text. <a href='http://x.com'>Click here.</a></p></body></html>"
        parser = HtmlParser.from_string(html, None, Tokenizer("english"))
        words = parser.stigma_words
        self.assertIsInstance(words, tuple)

    def test_stigma_words_fallback_on_empty(self):
        html = "<html><body><p>No links or strikethrough here.</p></body></html>"
        parser = HtmlParser.from_string(html, None, Tokenizer("english"))
        words = parser.stigma_words
        self.assertIsInstance(words, tuple)

    def test_document_exception_handler(self):
        html = "<html><body><p>Some content.</p></body></html>"
        parser = HtmlParser.from_string(html, None, Tokenizer("english"))
        with patch("sumy.parsers.html.lxml_html.fromstring", side_effect=Exception("parse error")):
            document = HtmlParser.document.func(parser)
        self.assertEqual(len(document.paragraphs), 0)

    def test_skip_tags_excluded(self):
        html = "<html><body><p>Real content.</p><script>var x=1;</script></body></html>"
        parser = HtmlParser.from_string(html, None, Tokenizer("english"))
        document = parser.document
        all_text = " ".join(str(s) for p in document.paragraphs for s in p.sentences)
        self.assertNotIn("var x", all_text)
