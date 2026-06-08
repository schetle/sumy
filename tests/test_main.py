import pytest
from io import StringIO
from click.testing import CliRunner
from sumy.__main__ import main, handle_arguments, AVAILABLE_METHODS


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "luhn" in result.output or "AVAILABLE_METHODS" in result.output or "lsa" in result.output


def test_valid_method_luhn():
    runner = CliRunner()
    result = runner.invoke(main, ["luhn"], input="Hello world. This is a test sentence.")
    assert result.exit_code == 0


def test_valid_method_lsa():
    runner = CliRunner()
    result = runner.invoke(main, ["lsa"], input="Hello world. This is a test sentence. And a third one.")
    assert result.exit_code == 0


def test_invalid_method():
    runner = CliRunner()
    result = runner.invoke(main, ["invalid-method"])
    assert result.exit_code != 0


def test_handle_arguments_default():
    summarizer, parser, items_count = handle_arguments(
        method="luhn",
        length="20%",
        language="english",
        default_input_stream=StringIO("Hello world. This is a test sentence.")
    )
    assert summarizer is not None
    assert parser is not None


def test_handle_arguments_wrong_format():
    with pytest.raises(ValueError):
        handle_arguments(
            method="lsa",
            length="20%",
            language="english",
            document_format="text",
            default_input_stream=StringIO("Whatever.")
        )


def test_all_methods_available():
    expected = {"luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"}
    assert set(AVAILABLE_METHODS.keys()) == expected
