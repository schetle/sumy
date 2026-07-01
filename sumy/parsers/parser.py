
from ..nlp.tokenizers import Tokenizer
class DocumentParser:
    """Abstract parser of input format into DOM."""

    SIGNIFICANT_WORDS: tuple[str, ...] = (
        "významný",
        "vynikající",
        "podstatný",
        "význačný",
        "důležitý",
        "slavný",
        "zajímavý",
        "eminentní",
        "vlivný",
        "supr",
        "super",
        "nejlepší",
        "dobrý",
        "kvalitní",
        "optimální",
        "relevantní",
    )
    STIGMA_WORDS: tuple[str, ...] = (
        "nejhorší",
        "zlý",
        "šeredný",
    )

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer: Tokenizer = tokenizer

    def tokenize_sentences(self, paragraph: object) -> tuple[str, ...]:
        return self._tokenizer.to_sentences(paragraph)

    def tokenize_words(self, sentence: object) -> tuple[str, ...]:
        return self._tokenizer.to_words(sentence)
