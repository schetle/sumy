# -*- coding: utf8 -*-

import sys
import urllib.request

import click

from itertools import chain
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
    "User-Agent": "Sumy (Automatic text summarizer) Version/%s" % __version__,
}
PARSERS = {
    "html": HtmlParser,
    "plaintext": PlaintextParser,
}


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
    ("Rouge-L (Summary Level)", False, rouge_l_summary_level)
)


@click.command()
@click.argument("method", type=click.Choice(list(AVAILABLE_METHODS)))
@click.argument("reference_summary", type=click.Path(exists=True))
@click.option("--length", default="20%", show_default=True,
              help="Length of summarized text.")
@click.option("--language", default="english", show_default=True,
              help="Natural language of summarized text.")
@click.option("--url", default=None, help="URL address of the page to summarize.")
@click.option("--file", "file_path", default=None,
              help="Path to the file to summarize.")
@click.option("--format", "document_format", default="plaintext",
              help="Format of input file.")
def main(method, reference_summary, length, language, url, file_path, document_format):
    """Sumy - evaluation of automatic text summarizer."""
    summarizer, document, items_count, reference = handle_arguments(
        method=method,
        reference_summary=reference_summary,
        length=length,
        language=language,
        url=url,
        file_path=file_path,
        document_format=document_format,
    )
    evaluated_sentences = summarizer(document, items_count)
    reference_document = PlaintextParser.from_string(reference, Tokenizer(language))
    reference_sentences = reference_document.document.sentences

    for name, evaluate_document, evaluate in AVAILABLE_EVALUATIONS:
        if evaluate_document:
            result = evaluate(evaluated_sentences, document.sentences)
        else:
            result = evaluate(evaluated_sentences, reference_sentences)
        print("%s: %f" % (name, result))


def handle_arguments(method, reference_summary, length, language,
                     url=None, file_path=None, document_format="plaintext"):
    if document_format not in PARSERS:
        raise ValueError("Unsupported format: %s. Possible: %s." % (
            document_format, ", ".join(PARSERS.keys())))

    if url is not None:
        parser_class = PARSERS["html"]
        request = urllib.request.Request(url, headers=HEADERS)
        input_stream = urllib.request.urlopen(request)
    elif file_path is not None:
        parser_class = PARSERS.get(document_format, PlaintextParser)
        input_stream = open(file_path, "rb")
    else:
        parser_class = PARSERS["plaintext"]
        input_stream = click.get_text_stream('stdin')

    items_count = ItemsCount(length)
    data = input_stream.read()
    if isinstance(data, bytes):
        data = data.decode('utf-8', errors='replace')
    parser = parser_class(data, Tokenizer(language))
    if hasattr(input_stream, 'close') and input_stream is not sys.stdin:
        input_stream.close()

    with open(reference_summary, "rb") as f:
        reference = f.read().decode("utf-8")

    summarizer = AVAILABLE_METHODS[method](parser, language)
    return summarizer, parser.document, items_count, reference


if __name__ == "__main__":
    main()
