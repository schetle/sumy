import pytest
from sumy.nlp.tokenizers import Tokenizer
from sumy.models.dom import Sentence
from .utils import build_document as _build_document
from .utils import build_document_from_string as _build_document_from_string
from .utils import load_resource as _load_resource


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
    return _build_document


@pytest.fixture
def build_document_from_string():
    return _build_document_from_string


@pytest.fixture
def load_resource():
    return _load_resource
