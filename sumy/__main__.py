# -*- coding: utf8 -*-

import sys
import argparse
import urllib.request

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


def main(args=None):
    parser = argparse.ArgumentParser(description="Automatic text summarizer")
    parser.add_argument("algorithm", choices=list(AVAILABLE_METHODS.keys()),
                        help="Summarization algorithm to use")
    parser.add_argument("--url", default=None,
                        help="URL address of the web page to summarize.")
    parser.add_argument("--file", default=None,
                        help="Path to the text file to summarize.")
    parser.add_argument("--length", default="20%",
                        help="Length of summarized text. It may be count of sentences "
                             "or percentage of input text. [default: 20%%]")
    parser.add_argument("--language", default="english",
                        help="Natural language of summarized text. [default: english]")
    parser.add_argument("--stopwords", default=None,
                        help="Path to a file containing a list of stopwords. "
                             "One word per line in UTF-8 encoding.")
    parser.add_argument("--format", default=None,
                        help="Format of input document. Possible values: html, plaintext")
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args(args)
    summarizer, parser, items_count = handle_arguments(args)

    for sentence in summarizer(parser.document, items_count):
        print(str(sentence))

    return 0


def handle_arguments(args, default_input_stream=sys.stdin):
    document_format = args.format
    if document_format is not None and document_format not in PARSERS:
        raise ValueError("Unsupported format of input document. Possible values are: %s. Given: %s." % (
            ", ".join(PARSERS.keys()),
            document_format,
        ))

    if args.url is not None:
        parser = PARSERS[document_format or "html"]
        request = urllib.request.Request(args.url, headers=HEADERS)
        input_stream = urllib.request.urlopen(request)
    elif args.file is not None:
        parser = PARSERS[document_format or "plaintext"]
        input_stream = open(args.file, "rb")
    else:
        parser = PARSERS[document_format or "plaintext"]
        input_stream = default_input_stream

    items_count = ItemsCount(args.length)

    language = args.language
    if args.stopwords:
        stop_words = read_stop_words(args.stopwords)
    else:
        stop_words = get_stop_words(language)

    parser = parser(input_stream.read(), Tokenizer(language))
    if input_stream is not sys.stdin:
        input_stream.close()

    stemmer = Stemmer(language)

    summarizer_class = AVAILABLE_METHODS[args.algorithm]
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
