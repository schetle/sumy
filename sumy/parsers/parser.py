"""Base class for document parsers."""


class DocumentParser:
    """Abstract parser of input format into DOM.

    Provides default significant and stigma words for parsers that don't
    extract them from document structure.
    """

    SIGNIFICANT_WORDS = (
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
    STIGMA_WORDS = (
        "nejhorší",
        "zlý",
        "šeredný",
    )

    def __init__(self, tokenizer):
        """Initialize a DocumentParser with a tokenizer.

        Args:
            tokenizer: Tokenizer instance for sentence and word splitting.
        """
        self._tokenizer = tokenizer

    def tokenize_sentences(self, paragraph: str) -> tuple:
        """Split a paragraph into sentences.

        Args:
            paragraph: Text to split into sentences.

        Returns:
            Tuple of sentence strings.
        """
        return self._tokenizer.to_sentences(paragraph)

    def tokenize_words(self, sentence: str) -> tuple:
        """Split a sentence into words.

        Args:
            sentence: Text to split into words.

        Returns:
            Tuple of word strings.
        """
        return self._tokenizer.to_words(sentence)
