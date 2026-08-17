import sys
import urllib.request

from itertools import chain

import click

from .. import __version__
from ..utils import ItemsCount, get_stop_words
from ..models import TfDocumentModel
from ..nlp.tokenizers import Tokenizer
from ..parsers.plaintext import PlaintextParser
from .._cli_common import HEADERS, PARSERS
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

def build_random(parser, language):
    return RandomSummarizer()


def build_luhn(parser, language):
    summarizer = LuhnSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_edmundson(parser, language):
    summarizer = EdmundsonSummarizer(Stemmer(language))
    summarizer.null_words = get_stop_words(language)
    summarizer.bonus_words = parser.significant_words
    summarizer.stigma_words = parser.stigma_words
    return summarizer


def build_lsa(parser, language):
    summarizer = LsaSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_text_rank(parser, language):
    summarizer = TextRankSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_lex_rank(parser, language):
    summarizer = LexRankSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_sum_basic(parser, language):
    summarizer = SumBasicSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def build_kl(parser, language):
    summarizer = KLSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    return summarizer


def evaluate_cosine_similarity(evaluated_sentences, reference_sentences):
    evaluated_words = tuple(chain(*(s.words for s in evaluated_sentences)))
    reference_words = tuple(chain(*(s.words for s in reference_sentences)))
    evaluated_model = TfDocumentModel(evaluated_words)
    reference_model = TfDocumentModel(reference_words)
    return cosine_similarity(evaluated_model, reference_model)


def evaluate_unit_overlap(evaluated_sentences, reference_sentences):
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


@click.command()
@click.argument("method", type=click.Choice(list(AVAILABLE_METHODS)))
@click.argument("reference_summary", type=click.Path(exists=True))
@click.option("--length", default="20%", show_default=True,
              help="Length of summarized text. Count of sentences or percentage of input text.")
@click.option("--language", default="english", show_default=True,
              help="Natural language of summarized text.")
@click.option("--format", "document_format", default=None, type=click.Choice(["html", "plaintext"]),
              help="Format of input document.")
@click.option("--url", default=None, help="URL of the web page to evaluate.")
@click.option("--file", "file_path", default=None, type=click.Path(exists=True),
              help="Path to the text file to evaluate.")
@click.version_option(version=__version__)
def main(method, reference_summary, length, language, document_format, url, file_path):
    """Sumy - evaluation of automatic text summary."""
    summarizer, document, items_count, ref_text = handle_arguments(
        method, reference_summary, length, language, document_format, url, file_path
    )

    evaluated_sentences = summarizer(document, items_count)
    reference_document = PlaintextParser.from_string(ref_text, Tokenizer(language))
    reference_sentences = reference_document.document.sentences

    for name, evaluate_document, evaluate in AVAILABLE_EVALUATIONS:
        if evaluate_document:
            result = evaluate(evaluated_sentences, document.sentences)
        else:
            result = evaluate(evaluated_sentences, reference_sentences)
        print("%s: %f" % (name, result))


def handle_arguments(method, reference_summary_path, length, language, document_format, url, file_path,
                     default_input_stream=None):
    if default_input_stream is None:
        default_input_stream = sys.stdin
    parser_cls = PARSERS["plaintext"]
    input_stream = default_input_stream

    if url is not None:
        parser_cls = PARSERS["html"]
        request = urllib.request.Request(url, headers=HEADERS)
        input_stream = urllib.request.urlopen(request)
    elif file_path is not None:
        parser_cls = PARSERS.get(document_format, PlaintextParser)
        input_stream = open(file_path, "rb")

    items_count = ItemsCount(length)

    try:
        content = input_stream.read()
    finally:
        if input_stream is not default_input_stream:
            input_stream.close()

    parser = parser_cls(content, Tokenizer(language))

    with open(reference_summary_path, "rb") as f:
        reference_text = f.read().decode("utf-8")

    summarizer_builder = AVAILABLE_METHODS[method]
    return summarizer_builder(parser, language), parser.document, items_count, reference_text


if __name__ == "__main__":
    main()
