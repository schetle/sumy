import os
import pytest
from sumy.nlp.tokenizers import Tokenizer
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence


@pytest.fixture(scope="session")
def tokenizer():
    return Tokenizer("czech")


@pytest.fixture
def build_sentence():
    def _build_sentence(sentence_as_string, is_heading=False, tok=None):
        if tok is None:
            tok = Tokenizer("czech")
        return Sentence(sentence_as_string, tok, is_heading)
    return _build_sentence


@pytest.fixture
def build_document():
    def _build_sentence(text, is_heading=False):
        return Sentence(text, Tokenizer("czech"), is_heading)

    def _build_document(*sets_of_sentences):
        paragraphs = []
        for sentences in sets_of_sentences:
            sentence_instances = []
            for s in sentences:
                if isinstance(s, str):
                    sentence_instances.append(_build_sentence(s))
                else:
                    sentence_instances.append(s)
            paragraphs.append(Paragraph(sentence_instances))
        return ObjectDocumentModel(paragraphs)

    return _build_document


@pytest.fixture
def build_document_from_string():
    def _build(string):
        sentences = []
        paragraphs = []
        tok = Tokenizer("czech")
        for line in string.strip().splitlines():
            line = line.lstrip()
            if line.startswith("# "):
                sentences.append(Sentence(line[2:], tok, is_heading=True))
            elif not line:
                if sentences:
                    paragraphs.append(Paragraph(sentences))
                sentences = []
            else:
                sentences.append(Sentence(line, tok))
        if sentences:
            paragraphs.append(Paragraph(sentences))
        return ObjectDocumentModel(paragraphs)

    return _build


@pytest.fixture
def load_resource():
    def _load(path):
        base = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
        full_path = os.path.join(base, path)
        with open(full_path, "rb") as f:
            return f.read().decode("utf-8")
    return _load
