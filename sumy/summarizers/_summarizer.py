"""Base class for all summarizer implementations."""

from collections import namedtuple
from operator import attrgetter
from ..utils import ItemsCount
from ..nlp.stemmers import null_stemmer


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating"))


class AbstractSummarizer:
    """Abstract base class for summarizers.

    Provides common methods for word normalization, stemming, and
    sentence ranking.
    """

    def __init__(self, stemmer=null_stemmer):
        """Initialize an AbstractSummarizer with a stemmer.

        Args:
            stemmer: Callable that stems a word. Defaults to null_stemmer.

        Raises:
            ValueError: If stemmer is not callable.
        """
        if not callable(stemmer):
            raise ValueError("Stemmer has to be a callable object")

        self._stemmer = stemmer

    def __call__(self, document, sentences_count):
        """Summarize a document by selecting the best sentences.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError("This method should be overridden in subclass")

    def stem_word(self, word: str) -> str:
        """Normalize and stem a word.

        Args:
            word: Word to stem.

        Returns:
            Stemmed word.
        """
        return self._stemmer(self.normalize_word(word))

    def normalize_word(self, word: str) -> str:
        """Normalize a word to lowercase.

        Args:
            word: Word to normalize.

        Returns:
            Lowercase word.
        """
        return str(word).lower()

    def _get_best_sentences(self, sentences, count, rating, *args, **kwargs):
        """Select the best sentences by rating, preserving document order.

        Args:
            sentences: All sentences from the document.
            count: Number of sentences to select (or ItemsCount).
            rating: Dict mapping sentences to ratings, or callable.
            *args: Extra args passed to rating callable.
            **kwargs: Extra kwargs passed to rating callable.

        Returns:
            Tuple of best sentences in document order.
        """
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        if not isinstance(count, ItemsCount):
            count = ItemsCount(count)
        infos = count(infos)
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos)
