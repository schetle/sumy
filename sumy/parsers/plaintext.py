from __future__ import annotations

from ..utils import cached_property
from ..models.dom import Sentence, Paragraph, ObjectDocumentModel
from ..nlp.tokenizers import Tokenizer
from .parser import DocumentParser


class PlaintextParser(DocumentParser):
    @classmethod
    def from_string(cls, string: str, tokenizer: Tokenizer) -> PlaintextParser:
        return cls(string, tokenizer)

    @classmethod
    def from_file(cls, file_path: str, tokenizer: Tokenizer) -> PlaintextParser:
        with open(file_path) as file:
            return cls(file.read(), tokenizer)

    def __init__(self, text: str, tokenizer: Tokenizer) -> None:
        super().__init__(tokenizer)
        self._text: str = (text.decode("utf-8") if isinstance(text, bytes) else str(text)).strip()

    @cached_property
    def significant_words(self) -> tuple[str, ...]:
        words: list[str] = []
        for paragraph in self.document.paragraphs:
            for heading in paragraph.headings:
                words.extend(heading.words)

        if words:
            return tuple(words)
        else:
            return self.SIGNIFICANT_WORDS

    @cached_property
    def stigma_words(self) -> tuple[str, ...]:
        return self.STIGMA_WORDS

    @cached_property
    def document(self) -> ObjectDocumentModel:
        current_paragraph: list[Sentence | str] = []
        paragraphs: list[Paragraph] = []
        for line in self._text.splitlines():
            line = line.strip()
            if line.isupper():
                heading = Sentence(line, self._tokenizer, is_heading=True)
                current_paragraph.append(heading)
            elif not line and current_paragraph:
                sentences = self._to_sentences(current_paragraph)
                paragraphs.append(Paragraph(sentences))
                current_paragraph = []
            elif line:
                current_paragraph.append(line)

        sentences = self._to_sentences(current_paragraph)
        paragraphs.append(Paragraph(sentences))

        return ObjectDocumentModel(paragraphs)

    def _to_sentences(self, lines: list[Sentence | str]) -> list[Sentence]:
        text = ""
        sentence_objects: list[Sentence] = []

        for line in lines:
            if isinstance(line, Sentence):
                if text:
                    sentences = self.tokenize_sentences(text)
                    sentence_objects += map(self._to_sentence, sentences)

                sentence_objects.append(line)
                text = ""
            else:
                text += " " + line

        text = text.strip()
        if text:
            sentences = self.tokenize_sentences(text)
            sentence_objects += map(self._to_sentence, sentences)

        return sentence_objects

    def _to_sentence(self, text: str) -> Sentence:
        assert text.strip()
        return Sentence(text, self._tokenizer)
