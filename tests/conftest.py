"""
Shared pytest fixtures and helpers for the sumy test suite.

Helper functions from tests/utils.py are exposed as pytest fixtures so that
test functions receive them via dependency injection instead of direct imports.
"""
import pytest
from .utils import build_document as _build_document
from .utils import build_document_from_string as _build_document_from_string
from .utils import build_sentence as _build_sentence
from .utils import load_resource as _load_resource
from .utils import expand_resource_path as _expand_resource_path


@pytest.fixture
def build_document():
    return _build_document


@pytest.fixture
def build_document_from_string():
    return _build_document_from_string


@pytest.fixture
def build_sentence():
    return _build_sentence


@pytest.fixture
def load_resource():
    return _load_resource


@pytest.fixture
def expand_resource_path():
    return _expand_resource_path
