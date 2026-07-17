from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from ...utils import _ensure_str

if TYPE_CHECKING:
    from ...nlp.tokenizers import Tokenizer


class Sentence(object):
    def __init__(self, text: str | bytes, tokenizer: "Tokenizer", is_heading: bool = False) -> None:
        self._text = _ensure_str(text).strip()
        self._tokenizer = tokenizer
        self._is_heading = bool(is_heading)

    @cached_property
    def words(self) -> tuple[str, ...]:
        return self._tokenizer.to_words(self._text)

    @property
    def is_heading(self) -> bool:
        return self._is_heading

    def __eq__(self, sentence: object) -> bool:
        assert isinstance(sentence, Sentence)
        return self._is_heading is sentence._is_heading and self._text == sentence._text

    def __ne__(self, sentence: object) -> bool:
        return not self.__eq__(sentence)

    def __hash__(self) -> int:
        return hash((self._is_heading, self._text))

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return "<%s: %s>" % (
            "Heading" if self._is_heading else "Sentence",
            self.__str__()
        )
