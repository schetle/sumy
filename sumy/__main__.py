"""Sumy - automatic text summarizer CLI."""

import argparse
import sys
from urllib import request as urllib

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
    "User-Agent": f"Sumy (Automatic text summarizer) Version/{__version__}",
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


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the sumy CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="sumy",
        description="Module for automatic summarization of text documents and HTML pages.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "method",
        choices=list(AVAILABLE_METHODS.keys()),
        help="Summarization method to use.",
    )
    parser.add_argument(
        "--length",
        default="20%",
        help="Length of summarized text. It may be count of sentences "
             "or percentage of input text. (default: 20%%)",
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Natural language of summarized text. (default: english)",
    )
    parser.add_argument(
        "--stopwords",
        default=None,
        help="Path to a file containing a list of stopwords. "
             "One word per line in UTF-8 encoding.",
    )
    parser.add_argument(
        "--format",
        default=None,
        dest="format",
        choices=["html", "plaintext"],
        help="Format of input document. Possible values: html, plaintext",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="URL address of the web page to summarize.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to the text file to summarize.",
    )
    return parser


def main(args=None):
    """Run the sumy summarizer CLI.

    Args:
        args: Command-line arguments. If None, reads from sys.argv.

    Returns:
        Exit code (0 for success).
    """
    parser = build_parser()
    parsed_args = parser.parse_args(args)
    summarizer, doc_parser, items_count = handle_arguments(parsed_args)

    for sentence in summarizer(doc_parser.document, items_count):
        print(str(sentence))

    return 0


def handle_arguments(args, default_input_stream=sys.stdin):
    """Parse and handle CLI arguments, returning summarizer, parser, and count.

    Args:
        args: Parsed argparse Namespace object.
        default_input_stream: Default input stream for stdin mode.

    Returns:
        Tuple of (summarizer, parser, items_count).

    Raises:
        ValueError: If an unsupported document format is given.
    """
    document_format = args.format

    if args.url is not None:
        parser_class = PARSERS[document_format or "html"]
        request = urllib.Request(args.url, headers=HEADERS)
        input_stream = urllib.urlopen(request)
    elif args.file is not None:
        parser_class = PARSERS[document_format or "plaintext"]
        input_stream = open(args.file, "rb")
    else:
        parser_class = PARSERS[document_format or "plaintext"]
        input_stream = default_input_stream

    items_count = ItemsCount(args.length)

    language = args.language
    if args.stopwords:
        stop_words = read_stop_words(args.stopwords)
    else:
        stop_words = get_stop_words(language)

    doc_parser = parser_class(input_stream.read(), Tokenizer(language))
    if input_stream is not sys.stdin:
        input_stream.close()

    stemmer = Stemmer(language)

    summarizer_class = AVAILABLE_METHODS[args.method]
    summarizer = build_summarizer(summarizer_class, stop_words, stemmer, doc_parser)

    return summarizer, doc_parser, items_count


def build_summarizer(summarizer_class, stop_words, stemmer, parser):
    """Build and configure a summarizer instance.

    Args:
        summarizer_class: Summarizer class to instantiate.
        stop_words: Stop words to set on the summarizer.
        stemmer: Stemmer to use.
        parser: Parser with significant/stigma words for Edmundson.

    Returns:
        Configured summarizer instance.
    """
    summarizer = summarizer_class(stemmer)
    if summarizer_class is EdmundsonSummarizer:
        summarizer.null_words = stop_words
        summarizer.bonus_words = parser.significant_words
        summarizer.stigma_words = parser.stigma_words
    else:
        summarizer.stop_words = stop_words
    return summarizer


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
