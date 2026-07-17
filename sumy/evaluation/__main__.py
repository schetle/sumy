import sys
from argparse import ArgumentParser
from itertools import chain
from urllib.request import Request, urlopen

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


def build_argument_parser():
    parser = ArgumentParser(
        description="Evaluation of automatic text summary.",
        prog="sumy_eval",
    )
    parser.add_argument("method", choices=list(AVAILABLE_METHODS.keys()),
                        help="Summarization algorithm to evaluate.")
    parser.add_argument("reference_summary",
                        help="Path to the file with reference summary.")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.add_argument("--length", default="20%",
                        help="Length of summarized text (sentence count or %% of input). [default: 20%%]")
    parser.add_argument("--language", default="english",
                        help="Natural language of summarized text. [default: english]")
    parser.add_argument("--format", choices=list(PARSERS.keys()),
                        help="Format of input document (html or plaintext). [default: plaintext]")
    parser.add_argument("--url", help="URL address of the web page to summarize.")
    parser.add_argument("--file", help="Path to the text file to summarize.")
    return parser


def main(args=None):
    parser = build_argument_parser()
    args = parser.parse_args(args)
    summarizer, document, items_count, reference_summary = handle_arguments(args)

    evaluated_sentences = summarizer(document, items_count)
    reference_document = PlaintextParser.from_string(reference_summary,
        Tokenizer(args.language))
    reference_sentences = reference_document.document.sentences

    for name, evaluate_document, evaluate in AVAILABLE_EVALUATIONS:
        if evaluate_document:
            result = evaluate(evaluated_sentences, document.sentences)
        else:
            result = evaluate(evaluated_sentences, reference_sentences)
        print("%s: %f" % (name, result))


def handle_arguments(args):
    document_format = args.format
    if document_format is not None and document_format not in PARSERS:
        raise ValueError("Unsupported format of input document. Possible values are: %s. Given: %s." % (
            ", ".join(PARSERS.keys()),
            document_format,
        ))

    parser_class = PARSERS["plaintext"]
    input_stream = sys.stdin

    if args.url is not None:
        parser_class = PARSERS["html"]
        request = Request(args.url, headers=HEADERS)
        input_stream = urlopen(request)
    elif args.file is not None:
        parser_class = PARSERS.get(document_format, PlaintextParser)
        input_stream = open(args.file, "rb")

    summarizer_builder = AVAILABLE_METHODS[args.method]

    items_count = ItemsCount(args.length)

    doc_parser = parser_class(input_stream.read(), Tokenizer(args.language))
    if input_stream is not sys.stdin:
        input_stream.close()

    with open(args.reference_summary, "rb") as file:
        reference_summary = file.read().decode("utf8")

    return summarizer_builder(doc_parser, args.language), doc_parser.document, items_count, reference_summary


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
