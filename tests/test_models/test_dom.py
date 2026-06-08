import pytest

from sumy.nlp.tokenizers import Tokenizer
from sumy.models.dom import Paragraph, Sentence
from ..utils import build_document, build_document_from_string


def test_unique_words():
    document = build_document(
        ("Nějaký muž šel kolem naší zahrady", "Nějaký muž šel kolem vaší zahrady",),
        ("Už už abych taky šel",),
    )

    returned = tuple(sorted(frozenset(document.words)))
    expected = (
        "Nějaký",
        "Už",
        "abych",
        "kolem",
        "muž",
        "naší",
        "taky",
        "už",
        "vaší",
        "zahrady",
        "šel"
    )
    assert expected == returned


def test_headings():
    document = build_document_from_string("""
        Nějaký muž šel kolem naší zahrady
        Nějaký jiný muž šel kolem vaší zahrady

        # Nová myšlenka
        Už už abych taky šel
    """)

    assert len(document.headings) == 1
    assert str(document.headings[0]) == "Nová myšlenka"


def test_sentences():
    document = build_document_from_string("""
        Nějaký muž šel kolem naší zahrady
        Nějaký jiný muž šel kolem vaší zahrady

        # Nová myšlenka
        Už už abych taky šel
    """)

    assert len(document.sentences) == 3
    assert str(document.sentences[0]) == "Nějaký muž šel kolem naší zahrady"
    assert str(document.sentences[1]) == "Nějaký jiný muž šel kolem vaší zahrady"
    assert str(document.sentences[2]) == "Už už abych taky šel"


def test_only_instances_of_sentence_allowed():
    document = build_document_from_string("""
        Nějaký muž šel kolem naší zahrady
        Nějaký jiný muž šel kolem vaší zahrady

        # Nová myšlenka
        Už už abych taky šel
    """)

    with pytest.raises(TypeError):
        Paragraph(list(document.sentences) + ["Last sentence"])


def test_sentences_equal():
    sentence1 = Sentence("", Tokenizer("czech"))
    sentence2 = Sentence("", Tokenizer("czech"))
    assert sentence1 == sentence2

    sentence1 = Sentence("word another.", Tokenizer("czech"))
    sentence2 = Sentence("word another.", Tokenizer("czech"))
    assert sentence1 == sentence2

    sentence1 = Sentence("word another", Tokenizer("czech"))
    sentence2 = Sentence("another word", Tokenizer("czech"))
    assert sentence1 != sentence2
