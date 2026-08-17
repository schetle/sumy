import pytest
from click.testing import CliRunner

from sumy.evaluation.__main__ import main, __version__


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def reference_file(tmp_path):
    ref = tmp_path / "reference.txt"
    ref.write_text(
        "Python is a programming language. It is used for data science.",
        encoding="utf-8",
    )
    return str(ref)


@pytest.fixture
def input_text():
    return (
        "Python is a high-level programming language. "
        "It was created by Guido van Rossum. "
        "Python is widely used in data science and web development. "
        "The language has a large standard library. "
        "Many developers prefer Python for its readability."
    )


def test_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Sumy" in result.output
    assert "METHOD" in result.output or "luhn" in result.output


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_valid_method_from_stdin(runner, reference_file, input_text):
    result = runner.invoke(
        main, ["lsa", reference_file, "--length", "2"], input=input_text
    )
    assert result.exit_code == 0
    assert "Precision:" in result.output or "Rouge" in result.output


def test_all_methods_accepted(runner, reference_file, input_text):
    methods = [
        "random", "luhn", "edmundson", "lsa",
        "text-rank", "lex-rank", "sum-basic", "kl",
    ]
    for method in methods:
        result = runner.invoke(
            main, [method, reference_file, "--length", "2"], input=input_text
        )
        assert result.exit_code == 0, (
            f"Method {method!r} failed with exit {result.exit_code}: {result.output}"
        )


def test_invalid_method(runner, reference_file):
    result = runner.invoke(main, ["invalid-method", reference_file])
    assert result.exit_code != 0


def test_invalid_format(runner, reference_file, input_text):
    result = runner.invoke(
        main, ["lsa", reference_file, "--format", "badformat"], input=input_text
    )
    assert result.exit_code != 0


def test_evaluation_output_contains_metrics(runner, reference_file, input_text):
    result = runner.invoke(
        main, ["luhn", reference_file, "--length", "2"], input=input_text
    )
    assert result.exit_code == 0
    assert "Precision:" in result.output
    assert "Recall:" in result.output
    assert "Rouge-1:" in result.output
