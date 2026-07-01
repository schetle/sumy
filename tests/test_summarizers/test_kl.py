import pytest

from sumy.models.dom._sentence import Sentence
from sumy.summarizers.kl import KLSummarizer
from ..utils import build_document
from sumy.nlp.tokenizers import Tokenizer


EMPTY_STOP_WORDS = []
COMMON_STOP_WORDS = ["the", "and", "i"]


def _build_summarizer(stop_words):
    summarizer = KLSummarizer()
    summarizer.stop_words = stop_words
    return summarizer


def test_empty_document():
    document = build_document()
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)

    returned = summarizer(document, 10)
    assert len(returned) == 0


def test_single_sentence():
    s = Sentence("I am one slightly longer sentence.", Tokenizer("english"))
    document = build_document([s])
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)

    returned = summarizer(document, 10)
    assert len(returned) == 1


def test_compute_word_freq():
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)

    words = ["one", "two", "three", "four"]
    freq = summarizer._compute_word_freq(words)
    assert freq.get("one", 0) == 1
    assert freq.get("two", 0) == 1
    assert freq.get("three", 0) == 1
    assert freq.get("four", 0) == 1

    words = ["one", "one", "two", "two"]
    freq = summarizer._compute_word_freq(words)
    assert freq.get("one", 0) == 2
    assert freq.get("two", 0) == 2
    assert freq.get("three", 0) == 0


def test_joint_freq():
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)
    w1 = ["one", "two", "three", "four"]
    w2 = ["one", "two", "three", "four"]
    freq = summarizer._joint_freq(w1, w2)
    assert freq["one"] == 1.0/4
    assert freq["two"] == 1.0/4
    assert freq["three"] == 1.0/4
    assert freq["four"] == 1.0/4

    w1 = ["one", "two", "three", "four"]
    w2 = ["one", "one", "three", "five"]
    freq = summarizer._joint_freq(w1, w2)
    assert freq["one"] == 3.0/8
    assert freq["two"] == 1.0/8
    assert freq["three"] == 1.0/4
    assert freq["four"] == 1.0/8
    assert freq["five"] == 1.0/8


def test_kl_divergence():
    summarizer = _build_summarizer(EMPTY_STOP_WORDS)

    w1 = {"one": .35, "two": .5, "three": .15}
    w2 = {"one": 1.0/3.0, "two": 1.0/3.0, "three": 1.0/3.0}

    kl_correct = 0.11475080798005841
    assert summarizer._kl_divergence(w1, w2) == pytest.approx(kl_correct, abs=1e-5)

    w1 = {"one": .1, "two": .2, "three": .7}
    w2 = {"one": .2, "two": .4, "three": .4}

    kl_correct = 0.1920419931617981
    assert summarizer._kl_divergence(w1, w2) == pytest.approx(kl_correct, abs=1e-5)
