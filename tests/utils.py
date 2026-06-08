from os.path import dirname, join, abspath
from sumy.nlp.tokenizers import Tokenizer
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence

_TOKENIZER = Tokenizer("czech")


def expand_resource_path(path):
    return join(abspath(dirname(__file__)), "data", path)


def load_resource(path):
    path = expand_resource_path(path)
    with open(path, "rb") as file:
        return file.read().decode('utf-8')


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
    for line in string.strip().splitlines():
        line = line.lstrip()
        if line.startswith("# "):
            sentences.append(Sentence(line[2:], _TOKENIZER, is_heading=True))
        elif not line:
            if sentences:
                paragraphs.append(Paragraph(sentences))
            sentences = []
        else:
            sentences.append(Sentence(line, _TOKENIZER))
    if sentences:
        paragraphs.append(Paragraph(sentences))
    return ObjectDocumentModel(paragraphs)


def build_sentence(sentence_as_string, is_heading=False):
    return Sentence(sentence_as_string, _TOKENIZER, is_heading)
