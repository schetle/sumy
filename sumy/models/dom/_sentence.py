from functools import cached_property
from ...utils import _ensure_str


class Sentence(object):
    def __init__(self, text, tokenizer, is_heading=False):
        self._text = _ensure_str(text).strip()
        self._tokenizer = tokenizer
        self._is_heading = bool(is_heading)

    @cached_property
    def words(self):
        return self._tokenizer.to_words(self._text)

    @property
    def is_heading(self):
        return self._is_heading

    def __eq__(self, sentence):
        assert isinstance(sentence, Sentence)
        return self._is_heading is sentence._is_heading and self._text == sentence._text

    def __ne__(self, sentence):
        return not self.__eq__(sentence)

    def __hash__(self):
        return hash((self._is_heading, self._text))

    def __str__(self):
        return self._text

    def __repr__(self):
        return "<%s: %s>" % (
            "Heading" if self._is_heading else "Sentence",
            self.__str__()
        )
