import pytest
from conftest import TXT
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.random import RandomSummarizer


@pytest.fixture(scope="module")
def doc():
    return PlaintextParser.from_file(TXT, Tokenizer("english")).document


def make(cls):
    s = cls(Stemmer("english"))
    s.stop_words = get_stop_words("english")
    return s


@pytest.mark.parametrize("cls", [LuhnSummarizer, LsaSummarizer, LexRankSummarizer, TextRankSummarizer,
                                 KLSummarizer, SumBasicSummarizer, RandomSummarizer])
def test_summarizer_returns_sentences(cls, doc):
    out = make(cls)(doc, 2)
    assert len(out) == 2
    assert all(str(s).strip() for s in out)


@pytest.mark.parametrize("method", ["cue_method", "key_method", "title_method", "location_method"])
def test_edmundson(method, doc):
    s = EdmundsonSummarizer(Stemmer("english"))
    s.bonus_words = ["the"]
    s.stigma_words = ["zzz"]
    s.null_words = ["a"]
    out = getattr(s, method)(doc, 2)
    assert 1 <= len(out) <= 2


def test_luhn_deterministic(doc):
    a = [str(x) for x in make(LuhnSummarizer)(doc, 2)]
    b = [str(x) for x in make(LuhnSummarizer)(doc, 2)]
    assert a == b


def test_percentage_length(doc):
    out = make(LsaSummarizer)(doc, "50%")
    assert 1 <= len(out) <= len(doc.sentences)
