import argparse
import sys

from urllib.request import Request, urlopen

from . import __version__
from .utils import ItemsCount, get_stop_words, read_stop_words
from .nlp.tokenizers import Tokenizer
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
def _get_html_parser():
    from .parsers.html import HtmlParser
    return HtmlParser


PARSERS = {
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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sumy",
        description="Sumy - automatic text summarizer.",
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
        help="Path to a file containing a list of stopwords. One word per line "
             "in UTF-8 encoding. If not provided, default list of stop-words "
             "is used according to chosen language.",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["html", "plaintext"],
        dest="format",
        help="Format of input document. Possible values: html, plaintext.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--url",
        default=None,
        help="URL address of the web page to summarize.",
    )
    source.add_argument(
        "--file",
        default=None,
        help="Path to the text file to summarize.",
    )
    return parser


def main(args=None):
    parser = build_parser()
    parsed_args = parser.parse_args(args)
    summarizer, doc_parser, items_count = handle_arguments(parsed_args)

    for sentence in summarizer(doc_parser.document, items_count):
        print(str(sentence))

    return 0


def _get_parser_class(format_name):
    if format_name == "html":
        return _get_html_parser()
    return PARSERS[format_name]


def handle_arguments(args, default_input_stream=sys.stdin):
    document_format = args.format
    valid_formats = list(PARSERS.keys()) + ["html"]
    if document_format is not None and document_format not in valid_formats:
        raise ValueError(
            f"Unsupported format of input document. "
            f"Possible values are: {', '.join(valid_formats)}. Given: {document_format}."
        )

    if args.url is not None:
        parser = _get_parser_class(document_format or "html")
        request = Request(args.url, headers=HEADERS)
        input_stream = urlopen(request)
    elif args.file is not None:
        parser = _get_parser_class(document_format or "plaintext")
        input_stream = open(args.file, "rb")
    else:
        parser = _get_parser_class(document_format or "plaintext")
        input_stream = default_input_stream

    items_count = ItemsCount(args.length)

    language = args.language
    if args.stopwords:
        stop_words = read_stop_words(args.stopwords)
    else:
        stop_words = get_stop_words(language)

    content = input_stream.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    parser = parser(content, Tokenizer(language))
    if input_stream is not sys.stdin:
        input_stream.close()

    stemmer = Stemmer(language)

    summarizer_class = AVAILABLE_METHODS[args.method]
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
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
