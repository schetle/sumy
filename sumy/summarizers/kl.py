"""KL-Divergence greedy summarization method."""

import math

from ._summarizer import AbstractSummarizer


class KLSummarizer(AbstractSummarizer):
    """KL-Sum: Greedily adds sentences to minimize KL Divergence.

    Source: http://www.aclweb.org/anthology/N09-1041
    """

    def __call__(self, document, sentences_count):
        """Summarize a document using KL-Divergence.

        Args:
            document: ObjectDocumentModel to summarize.
            sentences_count: Number of sentences to return.

        Returns:
            Tuple of best sentences.
        """
        ratings = self._get_ratings(document)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _get_ratings(self, document):
        """Compute ratings for all sentences.

        Args:
            document: ObjectDocumentModel to rate.

        Returns:
            Dict mapping sentences to ratings.
        """
        sentences = document.sentences
        ratings = self._compute_ratings(sentences)
        return ratings

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
        normalized_words = self._normalize_words(all_words)
        normalized_content_words = self._filter_out_stop_words(normalized_words)
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

    def _joint_freq(self, word_list_1, word_list_2):
        """Compute joint word frequency across two word lists.

        Args:
            word_list_1: First word list.
            word_list_2: Second word list.

        Returns:
            Dict mapping words to joint frequency.
        """
        # combined length of the word lists
        total_len = len(word_list_1) + len(word_list_2)

        # word frequencies within each list
        wc1 = self._compute_word_freq(word_list_1)
        wc2 = self._compute_word_freq(word_list_2)

        # inputs the counts from the first list
        joint = wc1.copy()

        # adds in the counts of the second list
        for k in wc2:
            if k in joint:
                joint[k] += wc2[k]
            else:
                joint[k] = wc2[k]

        # divides total counts by the combined length
        for k in joint:
            joint[k] /= float(total_len)

        return joint

    def _kl_divergence(self, summary_freq, doc_freq):
        """Compute KL Divergence between summary and document frequencies.

        Args:
            summary_freq: Summary word frequency distribution.
            doc_freq: Document word frequency distribution.

        Returns:
            KL Divergence value.
        """
        sum_val = 0
        for w in summary_freq:
            sum_val += doc_freq[w] * math.log(doc_freq[w] / summary_freq[w])
        return sum_val

    def _find_index_of_best_sentence(self, kls):
        """Find the sentence index with the smallest KL divergence.

        Args:
            kls: List of KL divergence values.

        Returns:
            Index of the best sentence.
        """
        return kls.index(min(kls))

    def _compute_ratings(self, sentences):
        """Compute ratings for all sentences using KL-Divergence algorithm.

        Args:
            sentences: Tuple of Sentence objects.

        Returns:
            Dict mapping sentences to ratings.
        """
        word_freq = self._compute_tf(sentences)
        ratings = {}
        summary = []

        # make it a list so that it can be modified
        sentences_list = list(sentences)

        # get all content words once for efficiency
        sentences_as_words = [self._get_content_words_in_sentence(s) for s in sentences]

        # Removes one sentence per iteration by adding to summary
        while len(sentences_list) > 0:
            # will store all the kls values for this pass
            kls = []

            # converts summary to normalized content word list
            summary_as_word_list = self._filter_out_stop_words(
                self._normalize_words(self._get_all_words_in_doc(summary))
            )

            for s in sentences_as_words:
                # calculates the joint frequency through combining the word lists
                joint_freq = self._joint_freq(s, summary_as_word_list)

                # adds the calculated kl divergence to the list
                kls.append(self._kl_divergence(joint_freq, word_freq))

            # find best sentence and add it into the summary
            index_to_remove = self._find_index_of_best_sentence(kls)
            best_sentence = sentences_list.pop(index_to_remove)
            del sentences_as_words[index_to_remove]
            summary.append(best_sentence)

            # value is the iteration in which it was removed multiplied by -1
            # so that the first sentences removed (the most important) have highest values
            ratings[best_sentence] = -1 * len(ratings)

        return ratings
