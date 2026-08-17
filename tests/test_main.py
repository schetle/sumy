import pytest
from click.testing import CliRunner
from sumy.__main__ import main, handle_arguments, __version__
from io import StringIO


@pytest.fixture
def runner():
    return CliRunner()


def test_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Sumy" in result.output
    assert "METHOD" in result.output or "luhn" in result.output


def test_version(runner):
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_valid_method_from_stdin(runner):
    text = (
        "Python is a high-level programming language. "
        "It was created by Guido van Rossum. "
        "Python is widely used in data science and web development."
    )
    result = runner.invoke(main, ["lsa", "--length", "1"], input=text)
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0


def test_invalid_method(runner):
    result = runner.invoke(main, ["invalid-method"])
    assert result.exit_code != 0


def test_invalid_format(runner):
    result = runner.invoke(main, ["lsa", "--format", "badformat"])
    assert result.exit_code != 0


def test_all_methods_accepted(runner):
    methods = ["luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"]
    text = (
        "Python is a high-level programming language. "
        "It was created by Guido van Rossum. "
        "Python is widely used in data science and web development. "
        "The language has a large standard library."
    )
    for method in methods:
        result = runner.invoke(main, [method, "--length", "1"], input=text)
        assert result.exit_code == 0, f"Method {method!r} failed: {result.output}"


def test_handle_arguments_plaintext():
    summarizer, parser, items_count = handle_arguments(
        method="lsa",
        length="20%",
        language="english",
        stopwords_path=None,
        document_format=None,
        url=None,
        file_path=None,
        default_input_stream=StringIO("Whatever sentence here."),
    )
    assert summarizer is not None
    assert parser is not None
    assert items_count is not None


def test_handle_arguments_wrong_format(runner):
    result = runner.invoke(main, ["lsa", "--format", "text"], input="Some text.")
    assert result.exit_code != 0


def test_handle_arguments_from_file(tmp_path):
    text_file = tmp_path / "doc.txt"
    text_file.write_bytes(
        b"Python is a high-level programming language. "
        b"It was created by Guido van Rossum. "
        b"Python is widely used in data science."
    )
    summarizer, parser, items_count = handle_arguments(
        method="lsa",
        length="1",
        language="english",
        stopwords_path=None,
        document_format=None,
        url=None,
        file_path=str(text_file),
    )
    assert summarizer is not None
    assert parser is not None
    assert items_count is not None


def test_handle_arguments_custom_stopwords(tmp_path):
    stopwords_file = tmp_path / "stopwords.txt"
    stopwords_file.write_bytes(b"the\na\nan\n")
    summarizer, parser, items_count = handle_arguments(
        method="lsa",
        length="1",
        language="english",
        stopwords_path=str(stopwords_file),
        document_format=None,
        url=None,
        file_path=None,
        default_input_stream=StringIO(
            "Python is a high-level programming language. "
            "It was created by Guido van Rossum."
        ),
    )
    assert summarizer is not None
    assert parser is not None
    assert items_count is not None
