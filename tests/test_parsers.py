import pytest

from sumy.parsers.plaintext import PlaintextParser
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from .utils import expand_resource_path


def test_html_significant_words():
    """significant_words must return non-empty tuple from real h1 content, not the class fallback."""
    path = expand_resource_path("snippets/paragraphs.html")
    url = "http://www.snippet.org/paragraphs.html"
    parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

    words = parser.significant_words
    assert isinstance(words, tuple)
    assert len(words) > 0
    assert words != HtmlParser.SIGNIFICANT_WORDS, (
        "significant_words returned the class-level fallback constant instead of words from the HTML"
    )
    # The h1 in paragraphs.html contains "Toto je nadpis prvej úrovne"
    assert any(w.lower() in ("toto", "nadpis", "prvej", "úrovne") for w in words)


def test_html_stigma_words():
    """stigma_words must return non-empty tuple from anchor/strike tags, not the class fallback.

    Uses a sufficiently long article so readability-lxml preserves inline tags.
    """
    html = """<html><head><title>Test Article</title></head><body>
<article>
<h1>Important Heading About Technology</h1>
<p>This is the main article text containing multiple sentences to ensure that
readability-lxml can extract the article content properly. The article discusses
various important topics that are relevant to the reader and provides context.</p>
<p>Here is another paragraph with a <a href="http://example.com">hyperlink destination</a>
and some <strike>outdated information</strike> that has been superseded by newer data.</p>
<p>Additional content to ensure sufficient article length for proper readability
extraction by the parser. This paragraph provides more text for the parser to work with
so that inline tags are preserved in the extracted summary output.</p>
</article>
</body></html>"""
    parser = HtmlParser(html.encode("utf-8"), Tokenizer("english"), "http://example.com")

    words = parser.stigma_words
    assert isinstance(words, tuple)
    assert len(words) > 0
    assert words != HtmlParser.STIGMA_WORDS, (
        "stigma_words returned the class-level fallback constant instead of words from the HTML"
    )
    # anchor or strike tag text should appear in the results
    assert any(w.lower() in ("hyperlink", "destination", "outdated", "information") for w in words)


def test_parse_plaintext():
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


def test_parse_plaintext_long():
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


def test_annotated_text():
    # readability-lxml produces separate paragraphs for headings and content
    # structure: para0=heading, para1=first-p, para2=second-p
    path = expand_resource_path("snippets/paragraphs.html")
    url = "http://www.snippet.org/paragraphs.html"
    parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

    document = parser.document

    assert len(document.paragraphs) == 3

    assert len(document.paragraphs[0].headings) == 1
    assert len(document.paragraphs[0].sentences) == 0

    assert str(document.paragraphs[0].headings[0]) == "Toto je nadpis prvej úrovne"

    assert len(document.paragraphs[1].headings) == 0
    assert len(document.paragraphs[1].sentences) == 1

    assert str(document.paragraphs[1].sentences[0]) == "Toto je prvý odstavec a to je fajn."

    assert len(document.paragraphs[2].headings) == 0
    assert len(document.paragraphs[2].sentences) == 2

    assert str(document.paragraphs[2].sentences[0]) == "Tento text je tu aby vyplnil prázdne miesto v srdci súboru."
    assert str(document.paragraphs[2].sentences[1]) == "Aj súbory majú predsa city."
