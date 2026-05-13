"""ROUGE evaluation metrics for automatic text summarization."""

from ..models.dom import Sentence


def _get_ngrams(n, text):
    """Compute n-grams from a text sequence.

    Args:
        n: Size of the n-grams.
        text: Sequence of words.

    Returns:
        Set of n-gram tuples.
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _split_into_words(sentences):
    """Split sentences into a flat list of words.

    Args:
        sentences: Iterable of Sentence objects.

    Returns:
        List of words from all sentences.

    Raises:
        ValueError: If any object is not a Sentence instance.
    """
    full_text_words = []
    for s in sentences:
        if not isinstance(s, Sentence):
            raise ValueError("Object in collection must be of type Sentence")
        full_text_words.extend(s.words)
    return full_text_words


def _get_word_ngrams(n, sentences):
    """Compute word n-grams from sentences.

    Args:
        n: Size of the n-grams.
        sentences: Iterable of Sentence objects.

    Returns:
        Set of word n-gram tuples.
    """
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


def _get_index_of_lcs(x, y):
    """Get the index dimensions for LCS computation.

    Args:
        x: First sequence.
        y: Second sequence.

    Returns:
        Tuple of (len(x), len(y)).
    """
    return len(x), len(y)


def _len_lcs(x, y):
    """Compute the length of the Longest Common Subsequence.

    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
        x: First sequence of words.
        y: Second sequence of words.

    Returns:
        Length of the LCS between x and y.
    """
    table = _lcs(x, y)
    n, m = _get_index_of_lcs(x, y)
    return table[n, m]


def _lcs(x, y):
    """Compute the LCS table using dynamic programming.

    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
        x: First sequence of words.
        y: Second sequence of words.

    Returns:
        Dict mapping (i, j) coordinates to LCS lengths.
    """
    n, m = _get_index_of_lcs(x, y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _recon_lcs(x, y):
    """Reconstruct the Longest Common Subsequence.

    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
        x: First sequence of words.
        y: Second sequence of words.

    Returns:
        Tuple of words in the LCS.
    """
    i, j = _get_index_of_lcs(x, y)
    table = _lcs(x, y)

    def _recon(i, j):
        """Recursively reconstruct the LCS.

        Args:
            i: Current index in x.
            j: Current index in y.

        Returns:
            List of (word, index) tuples.
        """
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(item[0] for item in _recon(i, j))
    return recon_tuple


def rouge_n(evaluated_sentences, reference_sentences, n=2):
    """Compute ROUGE-N of two text collections of sentences.

    Source: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf

    Args:
        evaluated_sentences: Sentences picked by the summarizer.
        reference_sentences: Sentences from the reference set.
        n: Size of n-gram. Defaults to 2.

    Returns:
        ROUGE-N score between 0.0 and 1.0.

    Raises:
        ValueError: If either collection is empty.
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    return overlapping_count / reference_count


def rouge_1(evaluated_sentences, reference_sentences):
    """Compute ROUGE-1 (unigram overlap).

    Args:
        evaluated_sentences: Sentences picked by the summarizer.
        reference_sentences: Sentences from the reference set.

    Returns:
        ROUGE-1 score between 0.0 and 1.0.
    """
    return rouge_n(evaluated_sentences, reference_sentences, 1)


def rouge_2(evaluated_sentences, reference_sentences):
    """Compute ROUGE-2 (bigram overlap).

    Args:
        evaluated_sentences: Sentences picked by the summarizer.
        reference_sentences: Sentences from the reference set.

    Returns:
        ROUGE-2 score between 0.0 and 1.0.
    """
    return rouge_n(evaluated_sentences, reference_sentences, 2)


def _f_lcs(llcs, m, n):
    """Compute the LCS-based F-measure score.

    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Args:
        llcs: Length of LCS.
        m: Number of words in reference summary.
        n: Number of words in candidate summary.

    Returns:
        LCS-based F-measure score.
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / r_lcs
    num = (1 + (beta ** 2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta ** 2) * p_lcs)
    return num / denom


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    """Compute ROUGE-L at sentence level.

    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Args:
        evaluated_sentences: Sentences picked by the summarizer.
        reference_sentences: Sentences from the reference set.

    Returns:
        ROUGE-L F-measure score.

    Raises:
        ValueError: If either collection is empty.
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = len(reference_words)
    n = len(evaluated_words)
    lcs = _len_lcs(evaluated_words, reference_words)
    return _f_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
    """Compute the union LCS score between a reference sentence and candidate summary.

    Args:
        evaluated_sentences: Sentences picked by the summarizer.
        reference_sentence: One sentence from the reference summaries.

    Returns:
        Union LCS value.

    Raises:
        ValueError: If evaluated_sentences is empty.
    """
    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = set()
    reference_words = _split_into_words([reference_sentence])
    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        evaluated_words = _split_into_words([eval_s])
        lcs = set(_recon_lcs(reference_words, evaluated_words))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    union_lcs_count = len(lcs_union)
    union_lcs_value = union_lcs_count / combined_lcs_length
    return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
    """Compute ROUGE-L at summary level.

    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf

    Args:
        evaluated_sentences: Sentences picked by the summarizer.
        reference_sentences: Sentences from the reference set.

    Returns:
        ROUGE-L F-measure score at summary level.

    Raises:
        ValueError: If either collection is empty.
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    # total number of words in reference sentences
    m = len(_split_into_words(reference_sentences))

    # total number of words in evaluated sentences
    n = len(_split_into_words(evaluated_sentences))

    union_lcs_sum_across_all_references = 0
    for ref_s in reference_sentences:
        union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences, ref_s)
    return _f_lcs(union_lcs_sum_across_all_references, m, n)
