"""Sumy - evaluation of automatic text summary.

Usage:
    sumy_eval (random | luhn | edmundson | lsa | text-rank | lex-rank | sum-basic | kl) <reference_summary> [--length=<length>] [--language=<lang>]
    sumy_eval (random | luhn | edmundson | lsa | text-rank | lex-rank | sum-basic | kl) <reference_summary> [--length=<length>] [--language=<lang>] --url=<url>
    sumy_eval (random | luhn | edmundson | lsa | text-rank | lex-rank | sum-basic | kl) <reference_summary> [--length=<length>] [--language=<lang>] --file=<file_path> --format=<file_format>
    sumy_eval --version
    sumy_eval --help

Options:
    <reference_summary>  Path to the file with reference summary.
    --url=<url>          URL address of summarized message.
    --file=<file>        Path to file with summarized text.
    --format=<format>    Format of input file. [default: plaintext]
    --length=<length>    Length of summarized text. It may be count of sentences
                         or percentage of input text. [default: 20%]
    --language=<lang>    Natural language of summarized text. [default: english]
    --version            Displays version of application.
    --help               Displays this text.

"""

import sys
from urllib import request as urllib

from itertools import chain
from docopt import docopt
from .. import __version__
from ..utils import ItemsCount, get_stop_words
from ..models import TfDocumentModel
from ..nlp.tokenizers import Tokenizer
from ..parsers.html import HtmlParser
from ..parsers.plaintext import PlaintextParser
from ..summarizers.random import RandomSummarizer
from ..summarizers.luhn import LuhnSummarizer
from ..summarizers.edmundson import EdmundsonSummarizer
from ..summarizers.lsa import LsaSummarizer
from ..summarizers.text_rank import TextRankSummarizer
from ..summarizers.lex_rank import LexRankSummarizer
from ..summarizers.sum_basic import SumBasicSummarizer
from ..summarizers.kl import KLSummarizer
from ..nlp.stemmers import Stemmer
from . import precision, recall, f_score, cosine_similarity, unit_overlap
from . import rouge_1, rouge_2, rouge_l_sentence_level, rouge_l_summary_level


HEADERS = {
    "User-Agent": f"Sumy (Automatic text summarizer) Version/{__version__}",
}
PARSERS = {
    "html": HtmlParser,
    "plaintext": PlaintextParser,
}


def build_random(parser, language):
    """Build a RandomSummarizer instance.

    Args:
        parser: Document parser (unused).
        language: Language name (unused).

    Returns:
        RandomSummarizer instance.
    """
    return RandomSummarizer()


def build_luhn(parser, language):
    """Build a LuhnSummarizer instance.

    Args:
        parser: Document parser (unused).
        language: Language name for stemmer and stop words.

    Returns:
        Configured LuhnSummarizer instance.
    """
    summarizer = LuhnSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_edmundson(parser, language):
    """Build an EdmundsonSummarizer instance.

    Args:
        parser: Document parser with significant/stigma words.
        language: Language name for stemmer and stop words.

    Returns:
        Configured EdmundsonSummarizer instance.
    """
    summarizer = EdmundsonSummarizer(Stemmer(language))
    summarizer.null_words = get_stop_words(language)
    summarizer.bonus_words = parser.significant_words
    summarizer.stigma_words = parser.stigma_words
    return summarizer


def build_lsa(parser, language):
    """Build an LsaSummarizer instance.

    Args:
        parser: Document parser (unused).
        language: Language name for stemmer and stop words.

    Returns:
        Configured LsaSummarizer instance.
    """
    summarizer = LsaSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_text_rank(parser, language):
    """Build a TextRankSummarizer instance.

    Args:
        parser: Document parser (unused).
        language: Language name for stemmer and stop words.

    Returns:
        Configured TextRankSummarizer instance.
    """
    summarizer = TextRankSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_lex_rank(parser, language):
    """Build a LexRankSummarizer instance.

    Args:
        parser: Document parser (unused).
        language: Language name for stemmer and stop words.

    Returns:
        Configured LexRankSummarizer instance.
    """
    summarizer = LexRankSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_sum_basic(parser, language):
    """Build a SumBasicSummarizer instance.

    Args:
        parser: Document parser (unused).
        language: Language name for stemmer and stop words.

    Returns:
        Configured SumBasicSummarizer instance.
    """
    summarizer = SumBasicSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_kl(parser, language):
    """Build a KLSummarizer instance.

    Args:
        parser: Document parser (unused).
        language: Language name for stemmer and stop words.

    Returns:
        Configured KLSummarizer instance.
    """
    summarizer = KLSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def evaluate_cosine_similarity(evaluated_sentences, reference_sentences):
    """Evaluate cosine similarity between evaluated and reference sentence sets.

    Args:
        evaluated_sentences: Sentences from the evaluated extract.
        reference_sentences: Sentences from the reference extract.

    Returns:
        Cosine similarity score.
    """
    evaluated_words = tuple(chain(*(s.words for s in evaluated_sentences)))
    reference_words = tuple(chain(*(s.words for s in reference_sentences)))
    evaluated_model = TfDocumentModel(evaluated_words)
    reference_model = TfDocumentModel(reference_words)

    return cosine_similarity(evaluated_model, reference_model)


def evaluate_unit_overlap(evaluated_sentences, reference_sentences):
    """Evaluate unit overlap between evaluated and reference sentence sets.

    Args:
        evaluated_sentences: Sentences from the evaluated extract.
        reference_sentences: Sentences from the reference extract.

    Returns:
        Unit overlap score.
    """
    evaluated_words = tuple(chain(*(s.words for s in evaluated_sentences)))
    reference_words = tuple(chain(*(s.words for s in reference_sentences)))
    evaluated_model = TfDocumentModel(evaluated_words)
    reference_model = TfDocumentModel(reference_words)

    return unit_overlap(evaluated_model, reference_model)


AVAILABLE_METHODS = {
    "random": build_random,
    "luhn": build_luhn,
    "edmundson": build_edmundson,
    "lsa": build_lsa,
    "text-rank": build_text_rank,
    "lex-rank": build_lex_rank,
    "sum-basic": build_sum_basic,
    "kl": build_kl,
}

AVAILABLE_EVALUATIONS = (
    ("Precision", False, precision),
    ("Recall", False, recall),
    ("F-score", False, f_score),
    ("Cosine similarity", False, evaluate_cosine_similarity),
    ("Cosine similarity (document)", True, evaluate_cosine_similarity),
    ("Unit overlap", False, evaluate_unit_overlap),
    ("Unit overlap (document)", True, evaluate_unit_overlap),
    ("Rouge-1", False, rouge_1),
    ("Rouge-2", False, rouge_2),
    ("Rouge-L (Sentence Level)", False, rouge_l_sentence_level),
    ("Rouge-L (Summary Level)", False, rouge_l_summary_level),
)


def main(args=None):
    """Run the sumy evaluation CLI.

    Args:
        args: Command-line arguments. If None, reads from sys.argv.
    """
    args = docopt(__doc__, args, version=__version__)
    summarizer, document, items_count, reference_summary = handle_arguments(args)

    evaluated_sentences = summarizer(document, items_count)
    reference_document = PlaintextParser.from_string(reference_summary,
        Tokenizer(args["--language"]))
    reference_sentences = reference_document.document.sentences

    for name, evaluate_document, evaluate in AVAILABLE_EVALUATIONS:
        if evaluate_document:
            result = evaluate(evaluated_sentences, document.sentences)
        else:
            result = evaluate(evaluated_sentences, reference_sentences)
        print(f"{name}: {result:f}")


def handle_arguments(args):
    """Parse and handle evaluation CLI arguments.

    Args:
        args: Parsed argument dict from docopt.

    Returns:
        Tuple of (summarizer, document, items_count, reference_summary).

    Raises:
        ValueError: If an unsupported document format is given.
    """
    document_format = args["--format"]
    if document_format is not None and document_format not in PARSERS:
        raise ValueError(
            f"Unsupported format of input document. "
            f"Possible values are: {', '.join(PARSERS.keys())}. Given: {document_format}."
        )

    parser = PARSERS["plaintext"]
    input_stream = sys.stdin

    if args["--url"] is not None:
        parser = PARSERS["html"]
        request = urllib.Request(args["--url"], headers=HEADERS)
        input_stream = urllib.urlopen(request)
    elif args["--file"] is not None:
        parser = PARSERS.get(document_format, PlaintextParser)
        input_stream = open(args["--file"], "rb")

    summarizer_builder = AVAILABLE_METHODS["luhn"]
    for method, builder in AVAILABLE_METHODS.items():
        if args[method]:
            summarizer_builder = builder
            break

    items_count = ItemsCount(args["--length"])

    parser = parser(input_stream.read(), Tokenizer(args["--language"]))
    if input_stream is not sys.stdin:
        input_stream.close()

    with open(args["<reference_summary>"], "rb") as file:
        reference_summary = file.read().decode("utf8")

    return summarizer_builder(parser, args["--language"]), parser.document, items_count, reference_summary


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
