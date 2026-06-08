# -*- coding: utf8 -*-

import sys
import urllib.request

import click

from . import __version__
from .utils import ItemsCount, get_stop_words, read_stop_words
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


@click.command()
@click.version_option(version=__version__, prog_name="sumy")
@click.argument("method", type=click.Choice(list(AVAILABLE_METHODS)))
@click.option("--length", default="20%", show_default=True,
              help="Length of summarized text. Count of sentences or percentage of input text.")
@click.option("--language", default="english", show_default=True,
              help="Natural language of summarized text.")
@click.option("--stopwords", "stopwords_path", default=None,
              help="Path to a file containing stop words (one per line, UTF-8).")
@click.option("--format", "document_format", default=None,
              help="Format of input document. Possible values: html, plaintext.")
@click.option("--url", default=None, help="URL address of the web page to summarize.")
@click.option("--file", "file_path", default=None, help="Path to the text file to summarize.")
def main(method, length, language, stopwords_path, document_format, url, file_path):
    """Sumy - automatic text summarizer."""
    summarizer, parser, items_count = handle_arguments(
        method=method,
        length=length,
        language=language,
        stopwords_path=stopwords_path,
        document_format=document_format,
        url=url,
        file_path=file_path,
        default_input_stream=click.get_text_stream('stdin'),
    )
    for sentence in summarizer(parser.document, items_count):
        print(str(sentence))


def handle_arguments(method, length, language, stopwords_path=None,
                     document_format=None, url=None, file_path=None,
                     default_input_stream=None):
    if default_input_stream is None:
        default_input_stream = sys.stdin
    if document_format is not None and document_format not in PARSERS:
        raise ValueError("Unsupported format. Possible: %s. Given: %s." % (
            ", ".join(PARSERS.keys()), document_format))

    if url is not None:
        parser_class = PARSERS[document_format or "html"]
        request = urllib.request.Request(url, headers=HEADERS)
        input_stream = urllib.request.urlopen(request)
    elif file_path is not None:
        parser_class = PARSERS[document_format or "plaintext"]
        input_stream = open(file_path, "rb")
    else:
        parser_class = PARSERS[document_format or "plaintext"]
        input_stream = default_input_stream

    items_count = ItemsCount(length)

    if stopwords_path:
        stop_words = read_stop_words(stopwords_path)
    else:
        stop_words = get_stop_words(language)

    data = input_stream.read()
    if isinstance(data, bytes):
        data = data.decode('utf-8', errors='replace')
    parser = parser_class(data, Tokenizer(language))
    if input_stream is not default_input_stream:
        input_stream.close()

    stemmer = Stemmer(language)
    summarizer_class = AVAILABLE_METHODS[method]
    summarizer = build_summarizer(summarizer_class, stop_words, stemmer, parser)

    return summarizer, parser, items_count


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
    main()
