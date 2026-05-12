import pytest

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from .utils import expand_resource_path


class TestParser:
    def test_parse_plaintext(self):
        parser = PlaintextParser.from_string("""
            Ako sa máš? Ja dobre! A ty? No
            mohlo to byť aj lepšie!!! Ale pohodička.


            TOTO JE AKOŽE NADPIS
            A toto je text pod ním, ktorý je textový.
            A tak ďalej...
        """, Tokenizer("czech"))

        document = parser.document

        assert len(document.paragraphs) == 2

        assert len(document.paragraphs[0].headings) == 0
        assert len(document.paragraphs[0].sentences) == 5

        assert len(document.paragraphs[1].headings) == 1
        assert len(document.paragraphs[1].sentences) == 2

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

        assert len(document.paragraphs) == 5

        assert len(document.paragraphs[0].headings) == 0
        assert len(document.paragraphs[0].sentences) == 5

        assert len(document.paragraphs[1].headings) == 1
        assert len(document.paragraphs[1].sentences) == 2

        assert len(document.paragraphs[2].headings) == 1
        assert len(document.paragraphs[2].sentences) == 3

        assert len(document.paragraphs[3].headings) == 0
        assert len(document.paragraphs[3].sentences) == 1

        assert len(document.paragraphs[4].headings) == 0
        assert len(document.paragraphs[4].sentences) == 1


def _breadability_available():
    try:
        from breadability.readable import Article  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _breadability_available(),
    reason="breadability not installed (HTML parser migration deferred to Milestone 2)"
)
class TestHtmlParser:
    def test_annotated_text(self):
        from sumy.parsers.html import HtmlParser

        path = expand_resource_path("snippets/paragraphs.html")
        url = "http://www.snippet.org/paragraphs.html"
        parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

        document = parser.document

        assert len(document.paragraphs) == 2

        assert len(document.paragraphs[0].headings) == 1
        assert len(document.paragraphs[0].sentences) == 1

        assert str(document.paragraphs[0].headings[0]) == "Toto je nadpis prvej úrovne"
        assert str(document.paragraphs[0].sentences[0]) == "Toto je prvý odstavec a to je fajn."

        assert len(document.paragraphs[1].headings) == 0
        assert len(document.paragraphs[1].sentences) == 2

        assert str(document.paragraphs[1].sentences[0]) == \
            "Tento text je tu aby vyplnil prázdne miesto v srdci súboru."
        assert str(document.paragraphs[1].sentences[1]) == "Aj súbory majú predsa city."
