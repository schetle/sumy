"""
Integration tests for sumy.__main__.main() CLI entry point.

These tests exercise the full CLI pipeline for each supported summarization
algorithm by calling main() with argparse-style argument lists and verifying
correct exit behaviour and non-empty output.
"""

import os
import pytest

from sumy.__main__ import main

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTICLES_DIR = os.path.join(DATA_DIR, "articles")

# A multi-sentence English plaintext article that all algorithms can process.
TEST_ARTICLE = os.path.join(ARTICLES_DIR, "test_article.txt")

# The Czech article is used for language-specific smoke tests.
CZECH_ARTICLE = os.path.join(ARTICLES_DIR, "prevko_cz_1.txt")


# ===========================================================================
# Argparse behaviour tests
# ===========================================================================


def test_help(capsys):
    """--help should print usage and exit with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["luhn", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "sumy" in captured.out.lower() or "usage" in captured.out.lower()


def test_version(capsys):
    """--version should print the version string and exit with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    # argparse prints version to stdout; the output should contain a version
    output = captured.out + captured.err
    assert output.strip() != ""


def test_invalid_method():
    """An unknown method name should cause argparse to exit with code 2."""
    with pytest.raises(SystemExit) as exc_info:
        main(["invalid-method"])
    assert exc_info.value.code == 2


# ===========================================================================
# Per-algorithm happy-path tests
# ===========================================================================

ALGORITHM_PARAMS = [
    "luhn",
    "edmundson",
    "lsa",
    "text-rank",
    "lex-rank",
    "sum-basic",
    "kl",
]


@pytest.mark.parametrize("algorithm", ALGORITHM_PARAMS)
def test_algorithm_produces_output(algorithm, capsys):
    """
    Each algorithm should run to completion (return 0) when given a
    plaintext file and a requested sentence count of 3.
    """
    result = main([
        algorithm,
        f"--file={TEST_ARTICLE}",
        "--format=plaintext",
        "--length=3",
    ])
    assert result == 0
    captured = capsys.readouterr()
    # At least one non-empty sentence must have been printed.
    assert captured.out.strip() != "", (
        f"Algorithm '{algorithm}' produced no output"
    )


def test_luhn_output_is_sentences(capsys):
    """Luhn output should consist of complete sentences (non-trivial check)."""
    result = main([
        "luhn",
        f"--file={TEST_ARTICLE}",
        "--format=plaintext",
        "--length=3",
    ])
    assert result == 0
    captured = capsys.readouterr()
    lines = [l.strip() for l in captured.out.splitlines() if l.strip()]
    assert len(lines) > 0, "Expected at least one output sentence from luhn"


# ---------------------------------------------------------------------------
# Edmundson requires significant_words / stigma_words to be available via the
# parser.  PlaintextParser exposes these as empty lists which is acceptable —
# the summariser simply has no bonus/stigma words and falls back to content
# scoring alone.
# ---------------------------------------------------------------------------


def test_edmundson_plaintext(capsys):
    """
    Edmundson with a plaintext file should succeed; bonus/stigma words come
    from the parser (empty lists for plaintext) and the summariser still runs.
    """
    result = main([
        "edmundson",
        f"--file={TEST_ARTICLE}",
        "--format=plaintext",
        "--length=3",
    ])
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out.strip() != "", "edmundson produced no output"


# ===========================================================================
# Length specification variants
# ===========================================================================


def test_length_as_count(capsys):
    """--length can be an integer sentence count."""
    result = main([
        "luhn",
        f"--file={TEST_ARTICLE}",
        "--format=plaintext",
        "--length=2",
    ])
    assert result == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    # We asked for 2 sentences; allow 0–2 (Edmundson can yield fewer).
    assert len(lines) <= 2


def test_length_as_percentage(capsys):
    """--length can be a percentage string like '10%'."""
    result = main([
        "luhn",
        f"--file={TEST_ARTICLE}",
        "--format=plaintext",
        "--length=10%",
    ])
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out.strip() != ""


# ===========================================================================
# Language option
# ===========================================================================


def test_language_option(capsys):
    """Specifying --language should not cause an error for a known language."""
    result = main([
        "luhn",
        f"--file={TEST_ARTICLE}",
        "--format=plaintext",
        "--length=3",
        "--language=english",
    ])
    assert result == 0


def test_czech_language_plaintext(capsys):
    """Czech language smoke test: luhn summarizer should produce output for Czech text."""
    result = main([
        "luhn",
        f"--file={CZECH_ARTICLE}",
        "--format=plaintext",
        "--length=3",
        "--language=czech",
    ])
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out.strip() != ""
