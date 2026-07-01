from sumy.parsers.plaintext import PlaintextParser
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from .utils import expand_resource_path


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
    path = expand_resource_path("snippets/paragraphs.html")
    url = "http://www.snippet.org/paragraphs.html"
    parser = HtmlParser.from_file(path, url, Tokenizer("czech"))

    document = parser.document

    assert len(document.paragraphs) == 2

    assert len(document.paragraphs[0].headings) == 1
    assert len(document.paragraphs[0].sentences) == 1

    assert str(document.paragraphs[0].headings[0]) == \
        "Toto je nadpis prvej úrovne"
    assert str(document.paragraphs[0].sentences[0]) == \
        "Toto je prvý odstavec a to je fajn."

    assert len(document.paragraphs[1].headings) == 0
    assert len(document.paragraphs[1].sentences) == 2

    assert str(document.paragraphs[1].sentences[0]) == \
        "Tento text je tu aby vyplnil prázdne miesto v srdci súboru."
    assert str(document.paragraphs[1].sentences[1]) == \
        "Aj súbory majú predsa city."


def test_html_from_string_document():
    html = "<html><body><p>First sentence. Second sentence.</p></body></html>"
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    document = parser.document
    assert len(document.paragraphs) >= 1
    assert any(len(p.sentences) > 0 for p in document.paragraphs)


def test_html_significant_words():
    html = (
        "<html><body>"
        "<p><strong>Important keyword</strong> and some filler text here.</p>"
        "</body></html>"
    )
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    words = parser.significant_words
    assert isinstance(words, tuple)
    assert len(words) > 0
    lower_words = [w.lower() for w in words]
    assert "important" in lower_words or "keyword" in lower_words


def test_html_significant_words_fallback():
    html = "<html><body><p>Plain text with no significant tags at all.</p></body></html>"
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    words = parser.significant_words
    assert words == HtmlParser.SIGNIFICANT_WORDS


def test_html_stigma_words():
    html = (
        "<html><body>"
        '<p>Normal text. <a href="http://x.com">click here for more</a> and after.</p>'
        "</body></html>"
    )
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    words = parser.stigma_words
    assert isinstance(words, tuple)
    assert len(words) > 0


def test_html_stigma_words_fallback():
    html = "<html><body><p>Plain text with no stigma tags at all.</p></body></html>"
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    words = parser.stigma_words
    assert words == HtmlParser.STIGMA_WORDS


def test_html_skip_tags_and_tail_text():
    html = (
        "<html><body>"
        "<p>Before code. <pre>ignored snippet</pre> After code.</p>"
        "</body></html>"
    )
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    document = parser.document
    all_text = " ".join(str(s) for p in document.paragraphs for s in p.sentences)
    assert "ignored snippet" not in all_text


def test_html_with_html_comments():
    html = (
        "<html><body>"
        "<!-- this is a comment -->"
        "<p>Sentence after comment.</p>"
        "</body></html>"
    )
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    document = parser.document
    assert len(document.paragraphs) >= 1
    all_text = " ".join(str(s) for p in document.paragraphs for s in p.sentences)
    assert "Sentence after comment" in all_text


def test_html_paragraphs_no_headings():
    html = (
        "<html><body>"
        "<p>First paragraph sentence one. First paragraph sentence two.</p>"
        "<p>Second paragraph sentence.</p>"
        "</body></html>"
    )
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    document = parser.document
    assert len(document.paragraphs) >= 1
    assert all(len(p.headings) == 0 for p in document.paragraphs)


def test_html_trailing_heading():
    html = (
        "<html><body>"
        "<p>Opening sentence here.</p>"
        "<h2>Trailing heading with no body</h2>"
        "</body></html>"
    )
    parser = HtmlParser.from_string(html, "http://example.com/", Tokenizer("english"))
    document = parser.document
    assert len(document.paragraphs) >= 1
    headings = [h for p in document.paragraphs for h in p.headings]
    assert len(headings) >= 1
