"""Shared test utilities for building documents, sentences, and loading resources."""

from io import StringIO
from os.path import dirname, join, abspath

from sumy.nlp.tokenizers import Tokenizer
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence


_TOKENIZER = Tokenizer("czech")


def expand_resource_path(path: str) -> str:
    """Expand a relative resource path to an absolute path within the test data directory.

    Args:
        path: Relative path within the test data directory.

    Returns:
        Absolute path to the resource.
    """
    return join(abspath(dirname(__file__)), "data", path)


def load_resource(path: str) -> str:
    """Load a test resource file as a string.

    Args:
        path: Relative path within the test data directory.

    Returns:
        Contents of the file as a string.
    """
    full_path = expand_resource_path(path)
    with open(full_path, "rb") as file:
        return file.read().decode("utf8")


def build_document(*sets_of_sentences):
    """Build an ObjectDocumentModel from groups of sentence strings.

    Args:
        *sets_of_sentences: Variable number of iterables, each containing
            sentence strings or Sentence objects for one paragraph.

    Returns:
        ObjectDocumentModel with the constructed paragraphs.
    """
    paragraphs = []
    for sentences in sets_of_sentences:
        sentence_instances = []
        for sentence_as_string in sentences:
            sentence = build_sentence(sentence_as_string)
            sentence_instances.append(sentence)

        paragraphs.append(Paragraph(sentence_instances))

    return ObjectDocumentModel(paragraphs)


def build_document_from_string(string: str):
    """Build an ObjectDocumentModel from a multi-line string with heading markers.

    Lines starting with '# ' are treated as headings. Blank lines separate paragraphs.

    Args:
        string: Multi-line text with optional '# ' heading markers.

    Returns:
        ObjectDocumentModel with the constructed paragraphs.
    """
    sentences = []
    paragraphs = []

    for line in string.strip().splitlines():
        line = line.lstrip()
        if line.startswith("# "):
            sentences.append(build_sentence(line[2:], is_heading=True))
        elif not line:
            paragraphs.append(Paragraph(sentences))
            sentences = []
        else:
            sentences.append(build_sentence(line))

    paragraphs.append(Paragraph(sentences))
    return ObjectDocumentModel(paragraphs)


def build_sentence(sentence_as_string, is_heading=False):
    """Build a Sentence object from a string.

    Args:
        sentence_as_string: Text of the sentence, or a Sentence object (returned as-is).
        is_heading: Whether this sentence is a heading.

    Returns:
        Sentence instance.
    """
    if isinstance(sentence_as_string, Sentence):
        return sentence_as_string
    return Sentence(sentence_as_string, _TOKENIZER, is_heading)
