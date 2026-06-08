import pytest
from sumy.nlp.tokenizers import Tokenizer
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence


@pytest.fixture(scope="session")
def tokenizer():
    return Tokenizer("czech")


def build_sentence(sentence_as_string, is_heading=False, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer("czech")
    return Sentence(sentence_as_string, tokenizer, is_heading)


def build_document(*sets_of_sentences):
    paragraphs = []
    for sentences in sets_of_sentences:
        sentence_instances = []
        for s in sentences:
            if isinstance(s, str):
                sentence_instances.append(build_sentence(s))
            else:
                sentence_instances.append(s)
        paragraphs.append(Paragraph(sentence_instances))
    return ObjectDocumentModel(paragraphs)


def build_document_from_string(string):
    sentences = []
    paragraphs = []
    tokenizer = Tokenizer("czech")
    for line in string.strip().splitlines():
        line = line.lstrip()
        if line.startswith("# "):
            sentences.append(Sentence(line[2:], tokenizer, is_heading=True))
        elif not line:
            if sentences:
                paragraphs.append(Paragraph(sentences))
            sentences = []
        else:
            sentences.append(Sentence(line, tokenizer))
    if sentences:
        paragraphs.append(Paragraph(sentences))
    return ObjectDocumentModel(paragraphs)
