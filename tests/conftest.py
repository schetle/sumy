"""
Shared pytest fixtures and helpers for the sumy test suite.

The module-level functions (build_document, build_document_from_string,
build_sentence, load_resource, expand_resource_path) mirror those in
tests/utils.py and are kept here as plain functions so that test files
can import them either way.  They are also exposed as pytest fixtures for
tests that prefer dependency injection.
"""
import pytest
from .utils import (
    build_document,
    build_document_from_string,
    build_sentence,
    load_resource,
    expand_resource_path,
)


@pytest.fixture
def build_document_fixture():
    return build_document


@pytest.fixture
def build_document_from_string_fixture():
    return build_document_from_string


@pytest.fixture
def load_resource_fixture():
    return load_resource
