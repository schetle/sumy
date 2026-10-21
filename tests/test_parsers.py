# -*- coding: utf8 -*-

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

    def test_parse_plaintext_from_bytes(self):
        # sumy's CLI (sumy/__main__.py) opens --file input in binary mode, so
        # PlaintextParser must decode bytes input instead of stringifying it
        # (str(b"...") would yield a "b'...'" repr, not the decoded text).
        parser = PlaintextParser.from_string(
            "Ako sa máš? Ja dobre!".encode("utf-8"),
            Tokenizer("czech"),
        )

        document = parser.document

        self.assertEqual(len(document.paragraphs), 1)
        self.assertEqual(
            str(document.paragraphs[0].sentences[0]), "Ako sa máš?"
        )


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

    def test_annotated_text_significant_and_stigma_words(self):
        # Explicit regression check for breadability's HTML extraction path
        # (significant_words/stigma_words rely on breadability's tag
        # annotations), so drift from breadability's old/stale lxml/chardet
        # pins is caught explicitly rather than only incidentally.
        path = expand_resource_path("snippets/paragraphs.html")
        url = "http://www.snippet.org/paragraphs.html"
        parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

        # The only heading in the fixture is an <h1>, which is a significant
        # tag, so significant_words is extracted from its text rather than
        # falling back to the parser's default word list.
        self.assertEqual(parser.significant_words,
            ("Toto", "je", "nadpis", "prvej", "úrovne"))

        # The fixture contains no <a>/<strike>/<s> annotated text, so
        # stigma_words falls back to the parser's default stigma words.
        self.assertEqual(parser.stigma_words, parser.STIGMA_WORDS)
