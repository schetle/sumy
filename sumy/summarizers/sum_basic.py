"""SumBasic frequency-based summarization method."""

from ._summarizer import AbstractSummarizer


class SumBasicSummarizer(AbstractSummarizer):
    """SumBasic: a frequency-based summarization system.

    Adjusts word frequencies as sentences are extracted.
    Source: http://www.cis.upenn.edu/~nenkova/papers/ipm.pdf
    """

    def __call__(self, document, sentences_count):
        """Summarize a document using SumBasic.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        sentences = document.sentences
        ratings = self._compute_ratings(sentences)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _get_all_words_in_doc(self, sentences):
        """Get all words from all sentences.

        Args:
            sentences: Iterable of Sentence objects.

        Returns:
            List of all words.
        """
        return [w for s in sentences for w in s.words]

    def _get_content_words_in_sentence(self, sentence):
        """Get normalized content words from a sentence.

        Args:
            sentence: Sentence to process.

        Returns:
            List of normalized content words (stop words filtered out).
        """
        normalized_words = self._normalize_words(sentence.words)
        normalized_content_words = self._filter_out_stop_words(normalized_words)
        return normalized_content_words

    def _normalize_words(self, words):
        """Normalize a list of words to lowercase.

        Args:
            words: List of words.

        Returns:
            List of normalized words.
        """
        return [self.normalize_word(w) for w in words]

    def _filter_out_stop_words(self, words):
        """Filter out stop words from a word list.

        Args:
            words: List of words.

        Returns:
            List of words without stop words.
        """
        return [w for w in words if w not in self.stop_words]

    def _compute_word_freq(self, list_of_words):
        """Compute word frequencies.

        Args:
            list_of_words: List of words.

        Returns:
            Dict mapping words to their frequency counts.
        """
        word_freq = {}
        for w in list_of_words:
            word_freq[w] = word_freq.get(w, 0) + 1
        return word_freq

    def _get_all_content_words_in_doc(self, sentences):
        """Get all normalized content words from all sentences.

        Args:
            sentences: Iterable of Sentence objects.

        Returns:
            List of normalized content words.
        """
        all_words = self._get_all_words_in_doc(sentences)
        content_words = self._filter_out_stop_words(all_words)
        normalized_content_words = self._normalize_words(content_words)
        return normalized_content_words

    def _compute_tf(self, sentences):
        """Compute the normalized term frequency.

        See: http://www.tfidf.com/

        Args:
            sentences: Iterable of Sentence objects.

        Returns:
            Dict mapping words to normalized term frequencies.
        """
        content_words = self._get_all_content_words_in_doc(sentences)
        content_words_count = len(content_words)
        content_words_freq = self._compute_word_freq(content_words)
        content_word_tf = {k: v / content_words_count for k, v in content_words_freq.items()}
        return content_word_tf

    def _compute_average_probability_of_words(self, word_freq_in_doc, content_words_in_sentence):
        """Compute the average word probability for a sentence.

        Args:
            word_freq_in_doc: Dict of word frequencies in the document.
            content_words_in_sentence: Content words in the sentence.

        Returns:
            Average word probability, or 0 if no content words.
        """
        content_words_count = len(content_words_in_sentence)
        if content_words_count > 0:
            word_freq_sum = sum(word_freq_in_doc[w] for w in content_words_in_sentence)
            word_freq_avg = word_freq_sum / content_words_count
            return word_freq_avg
        else:
            return 0

    def _update_tf(self, word_freq, words_to_update):
        """Update term frequencies by squaring them (probability update step).

        Args:
            word_freq: Dict of word frequencies.
            words_to_update: Words whose frequencies should be updated.

        Returns:
            Updated word frequency dict.
        """
        for w in words_to_update:
            word_freq[w] *= word_freq[w]
        return word_freq

    def _find_index_of_best_sentence(self, word_freq, sentences_as_words):
        """Find the sentence with the highest average word probability.

        Args:
            word_freq: Dict of word frequencies.
            sentences_as_words: List of word lists per sentence.

        Returns:
            Index of the best sentence.
        """
        min_possible_freq = -1
        max_value = min_possible_freq
        best_sentence_index = 0
        for i, words in enumerate(sentences_as_words):
            word_freq_avg = self._compute_average_probability_of_words(word_freq, words)
            if word_freq_avg > max_value:
                max_value = word_freq_avg
                best_sentence_index = i
        return best_sentence_index

    def _compute_ratings(self, sentences):
        """Compute ratings for all sentences using SumBasic algorithm.

        Args:
            sentences: Tuple of Sentence objects.

        Returns:
            Dict mapping sentences to ratings.
        """
        word_freq = self._compute_tf(sentences)
        ratings = {}

        # make it a list so that it can be modified
        sentences_list = list(sentences)

        # get all content words once for efficiency
        sentences_as_words = [self._get_content_words_in_sentence(s) for s in sentences]

        # Removes one sentence per iteration by adding to summary
        while len(sentences_list) > 0:
            best_sentence_index = self._find_index_of_best_sentence(word_freq, sentences_as_words)
            best_sentence = sentences_list.pop(best_sentence_index)

            # value is the iteration in which it was removed multiplied by -1
            # so that the first sentences removed (the most important) have highest values
            ratings[best_sentence] = -1 * len(ratings)

            # update probabilities
            best_sentence_words = sentences_as_words.pop(best_sentence_index)
            self._update_tf(word_freq, best_sentence_words)

        return ratings
