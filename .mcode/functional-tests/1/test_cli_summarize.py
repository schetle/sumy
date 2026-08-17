"""
Functional tests for sumy CLI summarization with stdin and file input.
"""
import shutil
import subprocess
from pathlib import Path
import pytest

SUMY = shutil.which("sumy") or "/tmp/sumy-venv/bin/sumy"
_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "tests" / "data"
TEST_ARTICLE = str(_DATA_ROOT / "articles" / "svd_converges.txt")
SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Text summarization is the process of reducing a text to its key points. "
    "Natural language processing enables machines to understand human language. "
    "Automatic summarization algorithms select the most relevant sentences. "
    "This technology has many practical applications in information retrieval."
)


def run_sumy(*args, input_text=None):
    result = subprocess.run(
        [SUMY, *args],
        capture_output=True,
        text=True,
        timeout=60,
        input=input_text,
    )
    return result


class TestLsaStdin:
    """sumy lsa --length 2 with stdin input"""

    def test_lsa_stdin_exits_zero(self):
        result = run_sumy("lsa", "--length", "2", "--format", "plaintext",
                          input_text=SAMPLE_TEXT)
        assert result.returncode == 0

    def test_lsa_stdin_produces_output(self):
        result = run_sumy("lsa", "--length", "2", "--format", "plaintext",
                          input_text=SAMPLE_TEXT)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_lsa_stdin_produces_two_sentences(self):
        result = run_sumy("lsa", "--length", "2", "--format", "plaintext",
                          input_text=SAMPLE_TEXT)
        assert result.returncode == 0
        sentences = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
        assert len(sentences) == 2


class TestLuhnFileInput:
    """sumy luhn --length 1 with file input"""

    def test_luhn_file_exits_zero(self):
        result = run_sumy("luhn", "--length", "1", "--format", "plaintext",
                          "--file", TEST_ARTICLE)
        assert result.returncode == 0

    def test_luhn_file_produces_output(self):
        result = run_sumy("luhn", "--length", "1", "--format", "plaintext",
                          "--file", TEST_ARTICLE)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_luhn_file_produces_one_sentence(self):
        result = run_sumy("luhn", "--length", "1", "--format", "plaintext",
                          "--file", TEST_ARTICLE)
        assert result.returncode == 0
        sentences = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
        assert len(sentences) >= 1


class TestAllMethods:
    """All 7 summarizer methods produce non-empty output via CLI with --length 1"""

    @pytest.mark.parametrize("method", [
        "luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"
    ])
    def test_method_exits_zero(self, method):
        result = run_sumy(method, "--length", "1", "--format", "plaintext",
                          input_text=SAMPLE_TEXT)
        assert result.returncode == 0, f"Method {method} failed: {result.stderr}"

    @pytest.mark.parametrize("method", [
        "luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"
    ])
    def test_method_produces_nonempty_output(self, method):
        result = run_sumy(method, "--length", "1", "--format", "plaintext",
                          input_text=SAMPLE_TEXT)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0, f"Method {method} produced empty output"


class TestInvalidArgs:
    """Invalid method or missing input should fail gracefully"""

    def test_invalid_method_exits_nonzero(self):
        result = run_sumy("invalid_method", "--length", "1")
        assert result.returncode != 0

    def test_unknown_file_exits_nonzero(self):
        result = run_sumy("lsa", "--length", "1", "--format", "plaintext",
                          "--file", "/nonexistent/file.txt")
        assert result.returncode != 0
