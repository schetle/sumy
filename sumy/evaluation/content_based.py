"""Content-based evaluation metrics: cosine similarity and unit overlap."""

from ..models import TfDocumentModel as TfModel


def cosine_similarity(evaluated_model, reference_model):
    """Compute cosine similarity of two TF document models.

    Args:
        evaluated_model: TfDocumentModel of the evaluated text.
        reference_model: TfDocumentModel of the reference text.

    Returns:
        Cosine similarity between 0.0 and 1.0.

    Raises:
        ValueError: If arguments are not TfDocumentModel instances or are empty.
    """
    if not (isinstance(evaluated_model, TfModel) and isinstance(reference_model, TfModel)):
        raise ValueError(
            "Arguments have to be instances of 'sumy.models.TfDocumentModel'")

    terms = frozenset(evaluated_model.terms) | frozenset(reference_model.terms)

    numerator = 0.0
    for term in terms:
        numerator += evaluated_model.term_frequency(term) * reference_model.term_frequency(term)

    denominator = evaluated_model.magnitude * reference_model.magnitude
    if denominator == 0.0:
        raise ValueError(f"Document model can't be empty. Given {evaluated_model!r} & {reference_model!r}")

    return numerator / denominator


def unit_overlap(evaluated_model, reference_model):
    """Compute unit overlap of two TF document models.

    Args:
        evaluated_model: TfDocumentModel of the evaluated text.
        reference_model: TfDocumentModel of the reference text.

    Returns:
        Unit overlap between 0.0 and 1.0.

    Raises:
        ValueError: If arguments are not TfDocumentModel instances or are empty.
    """
    if not (isinstance(evaluated_model, TfModel) and isinstance(reference_model, TfModel)):
        raise ValueError(
            "Arguments have to be instances of 'sumy.models.TfDocumentModel'")

    terms1 = frozenset(evaluated_model.terms)
    terms2 = frozenset(reference_model.terms)

    if not terms1 and not terms2:
        raise ValueError(
            "Documents can't be empty. Please pass the valid documents.")

    common_terms_count = len(terms1 & terms2)
    return common_terms_count / (len(terms1) + len(terms2) - common_terms_count)
