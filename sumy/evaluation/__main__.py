import sys

from itertools import chain
from typing import Optional
from urllib import request as urllib

import typer

from .. import __version__
from ..__main__ import _version_callback, HEADERS, PARSERS
from ..utils import ItemsCount, get_stop_words, validate_method
from ..models import TfDocumentModel
from ..nlp.tokenizers import Tokenizer
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

app = typer.Typer()


@app.command()
def main(
    method: str = typer.Argument(
        ..., help="Summarization method: random, luhn, edmundson, lsa, text-rank, lex-rank, sum-basic, kl"
    ),
    reference_summary: str = typer.Argument(..., help="Path to file with reference summary."),
    length: str = typer.Option("20%", help="Length of summarized text."),
    language: str = typer.Option("english", help="Natural language of summarized text."),
    url: Optional[str] = typer.Option(None, help="URL address of the web page to summarize."),
    file: Optional[str] = typer.Option(None, help="Path to file with summarized text."),
    format: Optional[str] = typer.Option(None, help="Format of input file: html or plaintext."),
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
):
    validate_method(method, AVAILABLE_METHODS)

    summarizer, document, items_count, ref_summary = handle_arguments(
        method=method,
        reference_summary=reference_summary,
        length=length,
        language=language,
        url=url,
        file=file,
        format=format,
    )

    evaluated_sentences = summarizer(document, items_count)
    reference_document = PlaintextParser.from_string(ref_summary, Tokenizer(language))
    reference_sentences = reference_document.document.sentences

    for name, evaluate_document, evaluate in AVAILABLE_EVALUATIONS:
        if evaluate_document:
            result = evaluate(evaluated_sentences, document.sentences)
        else:
            result = evaluate(evaluated_sentences, reference_sentences)
        print("%s: %f" % (name, result))


def handle_arguments(
    method: str,
    reference_summary: str,
    length: str = "20%",
    language: str = "english",
    url: Optional[str] = None,
    file: Optional[str] = None,
    format: Optional[str] = None,
    default_input_stream=None,
):
    if url is not None and file is not None:
        raise ValueError("Cannot specify both --url and --file. Use one or the other.")

    if format is not None and format not in PARSERS:
        raise ValueError(
            "Unsupported format of input document. Possible values are: %s. Given: %s." % (
                ", ".join(PARSERS.keys()),
                format,
            )
        )

    if default_input_stream is None:
        default_input_stream = sys.stdin

    parser_class = PARSERS["plaintext"]
    input_stream = default_input_stream

    if url is not None:
        parser_class = PARSERS["html"]
        req = urllib.Request(url, headers=HEADERS)
        input_stream = urllib.urlopen(req)
    elif file is not None:
        parser_class = PARSERS[format or "plaintext"]
        input_stream = open(file, "rb")

    summarizer_builder = AVAILABLE_METHODS[method]
    items_count = ItemsCount(length)

    content = input_stream.read()
    if isinstance(content, str):
        content = content.encode("utf-8")
    parser_obj = parser_class(content, Tokenizer(language))
    if input_stream is not default_input_stream:
        input_stream.close()

    with open(reference_summary, "rb") as f:
        ref_summary_text = f.read().decode("utf8")

    return summarizer_builder(parser_obj, language), parser_obj.document, items_count, ref_summary_text


if __name__ == "__main__":
    app()
