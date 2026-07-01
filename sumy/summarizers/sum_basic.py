
from ..models.dom import ObjectDocumentModel, Sentence
from ._summarizer import AbstractSummarizer
from ..utils import get_stop_words
class SumBasicSummarizer(AbstractSummarizer):
    """
    SumBasic: a frequency-based summarization system that adjusts word frequencies as
    sentences are extracted.
    Source: http://www.cis.upenn.edu/~nenkova/papers/ipm.pdf

    """

    def __call__(self, document: ObjectDocumentModel, sentences_count: int) -> tuple[Sentence, ...]:
        sentences = document.sentences
        ratings = self._compute_ratings(sentences)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _get_all_words_in_doc(self, sentences: tuple[Sentence, ...]) -> list[str]:
        return [w for s in sentences for w in s.words]

    def _get_content_words_in_sentence(self, sentence: Sentence) -> list[str]:
        normalized_words = self._normalize_words(sentence.words)
        normalized_content_words = self._filter_out_stop_words(normalized_words)
        return normalized_content_words

    def _normalize_words(self, words: tuple[str, ...] | list[str]) -> list[str]:
        return [self.normalize_word(w) for w in words]

    def _filter_out_stop_words(self, words: list[str]) -> list[str]:
        return [w for w in words if w not in self.stop_words]

    def _compute_word_freq(self, list_of_words: list[str]) -> dict[str, int]:
        word_freq: dict[str, int] = {}
        for w in list_of_words:
            word_freq[w] = word_freq.get(w, 0) + 1
        return word_freq

    def _get_all_content_words_in_doc(self, sentences: tuple[Sentence, ...]) -> list[str]:
        all_words = self._get_all_words_in_doc(sentences)
        content_words = self._filter_out_stop_words(all_words)
        normalized_content_words = self._normalize_words(content_words)
        return normalized_content_words

    def _compute_tf(self, sentences: tuple[Sentence, ...]) -> dict[str, float]:
        '''
        Computes the normalized term frequency as explained in http://www.tfidf.com/
        '''
        content_words = self._get_all_content_words_in_doc(sentences)
        content_words_count = len(content_words)
        content_words_freq = self._compute_word_freq(content_words)
        content_word_tf = dict((k, v / content_words_count) for (k, v) in content_words_freq.items())
        return content_word_tf

    def _compute_average_probability_of_words(self, word_freq_in_doc: dict[str, float],
                                              content_words_in_sentence: list[str]) -> float:
        content_words_count = len(content_words_in_sentence)
        if content_words_count > 0:
            word_freq_sum = sum([word_freq_in_doc[w] for w in content_words_in_sentence])
            word_freq_avg = word_freq_sum / content_words_count
            return word_freq_avg
        else:
            return 0

    def _update_tf(self, word_freq: dict[str, float], words_to_update: list[str]) -> dict[str, float]:
        for w in words_to_update:
            word_freq[w] *= word_freq[w]
        return word_freq
    def _find_index_of_best_sentence(self, word_freq: dict[str, float],
                                     sentences_as_words: list[list[str]]) -> int:
        min_possible_freq = -1
        max_value = min_possible_freq
        best_sentence_index = 0
        for i, words in enumerate(sentences_as_words):
            word_freq_avg = self._compute_average_probability_of_words(word_freq, words)
            if (word_freq_avg > max_value):
                max_value = word_freq_avg
                best_sentence_index = i
        return best_sentence_index
    def _compute_ratings(self, sentences: tuple[Sentence, ...]) -> dict[Sentence, int]:
        word_freq = self._compute_tf(sentences)
        ratings: dict[Sentence, int] = {}

        # make it a list so that it can be modified
        sentences_list = list(sentences)

        # get all content words once for efficiency
        sentences_as_words = [self._get_content_words_in_sentence(s) for s in sentences]

        # Removes one sentence per iteration by adding to summary
        while len(sentences_list) > 0:
            best_sentence_index = self._find_index_of_best_sentence(word_freq, sentences_as_words)
            best_sentence = sentences_list.pop(best_sentence_index)

            # value is the iteration in which it was removed multiplied by -1 so that the first sentences removed (the most important) have highest values
            ratings[best_sentence] =  -1 * len(ratings)

            # update probabilities
            best_sentence_words = sentences_as_words.pop(best_sentence_index)
            self._update_tf(word_freq, best_sentence_words)

        return ratings