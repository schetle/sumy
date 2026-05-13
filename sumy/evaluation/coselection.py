"""Coselection evaluation metrics: precision, recall, and F-score."""


def f_score(evaluated_sentences, reference_sentences, weight=1.0):
    """Compute the F-Score measure for extracted sentences.

    F(E) = ((W^2 + 1) * P(E) * R(E)) / (W^2 * P(E) + R(E))

    If W = 1.0 (default), computes the basic F-Score equivalent to
    (2 * P(E) * R(E)) / (P(E) + R(E)).

    Args:
        evaluated_sentences: Sentences of the evaluated extract.
        reference_sentences: Sentences of the reference extract.
        weight: Weighting factor favoring precision (W > 1) or recall (W < 1).

    Returns:
        F-Score value between 0.0 and 1.0.
    """
    p = precision(evaluated_sentences, reference_sentences)
    r = recall(evaluated_sentences, reference_sentences)

    weight **= 2  # weight = weight^2
    denominator = weight * p + r
    if denominator == 0.0:
        return 0.0
    else:
        return ((weight + 1) * p * r) / denominator


def precision(evaluated_sentences, reference_sentences):
    """Compute precision for extracted sentences.

    P(E) = A / B, where A is the count of common sentences and B is
    the count of sentences in the evaluated extract.

    Args:
        evaluated_sentences: Sentences of the evaluated extract.
        reference_sentences: Sentences of the reference extract.

    Returns:
        Precision value between 0.0 and 1.0.

    Raises:
        ValueError: If either collection is empty.
    """
    return _divide_evaluation(reference_sentences, evaluated_sentences)


def recall(evaluated_sentences, reference_sentences):
    """Compute recall for extracted sentences.

    R(E) = A / C, where A is the count of common sentences and C is
    the count of sentences in the reference extract.

    Args:
        evaluated_sentences: Sentences of the evaluated extract.
        reference_sentences: Sentences of the reference extract.

    Returns:
        Recall value between 0.0 and 1.0.

    Raises:
        ValueError: If either collection is empty.
    """
    return _divide_evaluation(evaluated_sentences, reference_sentences)


def _divide_evaluation(numerator_sentences, denominator_sentences):
    """Compute the ratio of common sentences to denominator size.

    Args:
        numerator_sentences: Sentences contributing to the numerator.
        denominator_sentences: Sentences contributing to the denominator.

    Returns:
        Ratio of common sentences to denominator size.

    Raises:
        ValueError: If either collection is empty.
    """
    denominator_sentences = frozenset(denominator_sentences)
    numerator_sentences = frozenset(numerator_sentences)

    if len(numerator_sentences) == 0 or len(denominator_sentences) == 0:
        raise ValueError("Both collections have to contain at least 1 sentence.")

    common_count = len(denominator_sentences & numerator_sentences)
    choosen_count = len(denominator_sentences)

    assert choosen_count != 0
    return common_count / choosen_count
