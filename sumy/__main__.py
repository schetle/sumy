"""Sumy - automatic text summarizer.

Usage:
    sumy (luhn | edmundson | lsa | text-rank | lex-rank | sum-basic | kl) [--length=<length>] [--language=<lang>] [--stopwords=<file_path>] [--format=<format>]
    sumy (luhn | edmundson | lsa | text-rank | lex-rank | sum-basic | kl) [--length=<length>] [--language=<lang>] [--stopwords=<file_path>] [--format=<format>] --url=<url>
    sumy (luhn | edmundson | lsa | text-rank | lex-rank | sum-basic | kl) [--length=<length>] [--language=<lang>] [--stopwords=<file_path>] [--format=<format>] --file=<file_path>
    sumy --version
    sumy --help

Options:
    --length=<length>        Length of summarized text. It may be count of sentences
                             or percentage of input text. [default: 20%]
    --language=<lang>        Natural language of summarized text. [default: english]
    --stopwords=<file_path>  Path to a file containing a list of stopwords. One word per line in UTF-8 encoding.
                             If it's not provided default list of stop-words is used according to chosen language.
    --format=<format>        Format of input document. Possible values: html, plaintext
    --url=<url>              URL address of the web page to summarize.
    --file=<file_path>       Path to the text file to summarize.
    --version                Displays current application version.
    --help                   Displays this text.

"""

import sys
from urllib import request as urllib

from docopt import docopt
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


def main(args=None):
    """Run the sumy summarizer CLI.

    Args:
        args: Command-line arguments. If None, reads from sys.argv.

    Returns:
        Exit code (0 for success).
    """
    args = docopt(__doc__, args, version=__version__)
    summarizer, parser, items_count = handle_arguments(args)

    for sentence in summarizer(parser.document, items_count):
        print(str(sentence))

    return 0


def handle_arguments(args, default_input_stream=sys.stdin):
    """Parse and handle CLI arguments, returning summarizer, parser, and count.

    Args:
        args: Parsed argument dict from docopt.
        default_input_stream: Default input stream for stdin mode.

    Returns:
        Tuple of (summarizer, parser, items_count).

    Raises:
        ValueError: If an unsupported document format is given.
    """
    document_format = args["--format"]
    if document_format is not None and document_format not in PARSERS:
        raise ValueError(
            f"Unsupported format of input document. "
            f"Possible values are: {', '.join(PARSERS.keys())}. Given: {document_format}."
        )

    if args["--url"] is not None:
        parser = PARSERS[document_format or "html"]
        request = urllib.Request(args["--url"], headers=HEADERS)
        input_stream = urllib.urlopen(request)
    elif args["--file"] is not None:
        parser = PARSERS[document_format or "plaintext"]
        input_stream = open(args["--file"], "rb")
    else:
        parser = PARSERS[document_format or "plaintext"]
        input_stream = default_input_stream

    items_count = ItemsCount(args["--length"])

    language = args["--language"]
    if args["--stopwords"]:
        stop_words = read_stop_words(args["--stopwords"])
    else:
        stop_words = get_stop_words(language)

    parser = parser(input_stream.read(), Tokenizer(language))
    if input_stream is not sys.stdin:
        input_stream.close()

    stemmer = Stemmer(language)

    summarizer_class = next(cls for name, cls in AVAILABLE_METHODS.items() if args[name])
    summarizer = build_summarizer(summarizer_class, stop_words, stemmer, parser)

    return summarizer, parser, items_count


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
