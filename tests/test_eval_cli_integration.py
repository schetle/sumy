"""
Integration tests for sumy.evaluation.__main__.main() CLI entry point.

These tests exercise the evaluation CLI (sumy_eval) for each supported
summarization algorithm, comparing against a short reference summary file.
"""

import os
import pytest

from sumy.evaluation.__main__ import main

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTICLES_DIR = os.path.join(DATA_DIR, "articles")

TEST_ARTICLE = os.path.join(ARTICLES_DIR, "test_article.txt")
REFERENCE_SUMMARY = os.path.join(ARTICLES_DIR, "test_reference_summary.txt")


# ===========================================================================
# Argparse behaviour tests
# ===========================================================================


def test_help(capsys):
    """--help should print usage and exit with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        # reference_summary is a required positional argument so argparse
        # will process --help before complaining about missing positionals.
        main(["luhn", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert output.strip() != ""


def test_version(capsys):
    """--version should exit 0 and print a non-empty version string."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert output.strip() != ""


# ===========================================================================
# Per-algorithm happy-path tests
# ===========================================================================

ALGORITHM_PARAMS = [
    "random",
    "luhn",
    "lsa",
    "text-rank",
    "lex-rank",
    "sum-basic",
    "kl",
]


@pytest.mark.parametrize("algorithm", ALGORITHM_PARAMS)
def test_algorithm_evaluation_output(algorithm, capsys):
    """
    Each algorithm should run to completion when given an article file and a
    reference summary, printing evaluation metric scores to stdout.
    """
    result = main([
        algorithm,
        REFERENCE_SUMMARY,
        f"--file={TEST_ARTICLE}",
        "--format=plaintext",
        "--length=3",
    ])
    assert result == 0
    captured = capsys.readouterr()
    # The evaluation CLI prints lines like "Precision: 0.250000"
    output = captured.out.strip()
    assert output != "", (
        f"Evaluation with '{algorithm}' produced no output"
    )
    # Verify at least one metric line is present.
    assert any(":" in line for line in output.splitlines()), (
        f"No metric lines found in output for '{algorithm}'"
    )


def test_edmundson_evaluation(capsys):
    """
    Edmundson evaluation should succeed with plaintext format (bonus/stigma
    words sourced from the parser, empty lists acceptable).
    """
    result = main([
        "edmundson",
        REFERENCE_SUMMARY,
        f"--file={TEST_ARTICLE}",
        "--format=plaintext",
        "--length=3",
    ])
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out.strip() != ""
