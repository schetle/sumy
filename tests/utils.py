from pathlib import Path

from sumy.nlp.tokenizers import Tokenizer
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence


_TOKENIZER = Tokenizer("czech")


def expand_resource_path(path):
    return str(Path(__file__).parent / "data" / path)


def load_resource(path):
    resource_path = expand_resource_path(path)
    with open(resource_path, "rb") as file:
        return file.read().decode("utf-8")


def build_document(*sets_of_sentences):
    paragraphs = []
    for sentences in sets_of_sentences:
        sentence_instances = []
        for sentence_as_string in sentences:
            sentence = build_sentence(sentence_as_string)
            sentence_instances.append(sentence)

        paragraphs.append(Paragraph(sentence_instances))

    return ObjectDocumentModel(paragraphs)


def build_document_from_string(string):
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
    return Sentence(sentence_as_string, _TOKENIZER, is_heading)
