import sys
from typing import Optional
from urllib import request as urllib

import typer

from . import __version__
from .utils import ItemsCount, get_stop_words, read_stop_words, read_stream_as_bytes, validate_method
from .nlp.tokenizers import Tokenizer
from .parsers.html import HtmlParser
from .parsers.plaintext import PlaintextParser
from .summarizers.luhn import LuhnSummarizer
from .summarizers.edmundson import EdmundsonSummarizer
from .summarizers.lsa import LsaSummarizer
from .summarizers.text_rank import TextRankSummarizer
from .summarizers.lex_rank import LexRankSummarizer
from .summarizers.sum_basic import SumBasicSummarizer
from .summarizers.kl import KLSummarizer
from .nlp.stemmers import Stemmer

HEADERS = {
    "User-Agent": "Sumy (Automatic text summarizer) Version/%s" % __version__,
}
PARSERS = {
    "html": HtmlParser,
    "plaintext": PlaintextParser,
}
AVAILABLE_METHODS = {
    "luhn": LuhnSummarizer,
    "edmundson": EdmundsonSummarizer,
    "lsa": LsaSummarizer,
    "text-rank": TextRankSummarizer,
    "lex-rank": LexRankSummarizer,
    "sum-basic": SumBasicSummarizer,
    "kl": KLSummarizer,
}

app = typer.Typer()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def main(
    method: str = typer.Argument(
        ..., help="Summarization method: luhn, edmundson, lsa, text-rank, lex-rank, sum-basic, kl"
    ),
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
    length: str = typer.Option("20%", help="Length of summarized text (count of sentences or percentage)."),
    language: str = typer.Option("english", help="Natural language of summarized text."),
    stopwords: Optional[str] = typer.Option(None, help="Path to stopwords file (one word per line, UTF-8)."),
    format: Optional[str] = typer.Option(None, help="Format of input document: html or plaintext."),
    url: Optional[str] = typer.Option(None, help="URL address of the web page to summarize."),
    file: Optional[str] = typer.Option(None, help="Path to the text file to summarize."),
):
    validate_method(method, AVAILABLE_METHODS)

    try:
        summarizer, parser, items_count = handle_arguments(
            method=method,
            length=length,
            language=language,
            stopwords=stopwords,
            format=format,
            url=url,
            file=file,
        )
        for sentence in summarizer(parser.document, items_count):
            print(str(sentence))
    except KeyboardInterrupt:
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


def handle_arguments(
    method: str,
    length: str = "20%",
    language: str = "english",
    stopwords: Optional[str] = None,
    format: Optional[str] = None,
    url: Optional[str] = None,
    file: Optional[str] = None,
    default_input_stream=None,
):
    if default_input_stream is None:
        default_input_stream = sys.stdin

    if url is not None and file is not None:
        raise ValueError("Cannot specify both --url and --file. Use one or the other.")

    if format is not None and format not in PARSERS:
        raise ValueError(
            "Unsupported format of input document. Possible values are: %s. Given: %s." % (
                ", ".join(PARSERS.keys()),
                format,
            )
        )

    if url is not None:
        parser_class = PARSERS[format or "html"]
        req = urllib.Request(url, headers=HEADERS)
        input_stream = urllib.urlopen(req)
    elif file is not None:
        parser_class = PARSERS[format or "plaintext"]
        input_stream = open(file, "rb")
    else:
        parser_class = PARSERS[format or "plaintext"]
        input_stream = default_input_stream

    items_count = ItemsCount(length)

    if stopwords:
        stop_words = read_stop_words(stopwords)
    else:
        stop_words = get_stop_words(language)

    content = read_stream_as_bytes(input_stream)
    parser_obj = parser_class(content, Tokenizer(language))
    if input_stream is not default_input_stream:
        input_stream.close()

    stemmer = Stemmer(language)
    summarizer_class = AVAILABLE_METHODS[method]
    summarizer = build_summarizer(summarizer_class, stop_words, stemmer, parser_obj)

    return summarizer, parser_obj, items_count


def build_summarizer(summarizer_class, stop_words, stemmer, parser):
    summarizer = summarizer_class(stemmer)
    if summarizer_class is EdmundsonSummarizer:
        summarizer.null_words = stop_words
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
    else:
        summarizer.stop_words = stop_words
    return summarizer


if __name__ == "__main__":
    app()
