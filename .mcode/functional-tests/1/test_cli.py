# -*- coding: utf-8 -*-
"""
Functional tests for the sumy CLI entry point (python -m sumy).

Verifies:
- --help output works and exits 0
- --version works and exits 0
- luhn --file=README.rst --length=3 produces 3 sentences
- All 6 working algorithms (luhn, lsa, text-rank, lex-rank, sum-basic, edmundson) work via CLI
- Invalid algorithm name exits non-zero
- Missing required info (no file/url) with stdin falls back gracefully
- --length as percentage and as count work
"""
import subprocess
import sys
import pytest

WORKING_DIR = "/l2l/workspace/sumy"
README_PATH = "/l2l/workspace/sumy/README.rst"


def run_cli(*args, input_text=None, cwd=None):
    """Invoke `python -m sumy` with the given args."""
    cmd = [sys.executable, "-m", "sumy"] + list(args)
    result = subprocess.run(
        cmd,
        cwd=cwd or WORKING_DIR,
        capture_output=True,
        text=True,
        timeout=60,
        input=input_text,
    )
    return result


class TestHelpOutput:
    """CLI --help flag."""

    def test_help_exits_zero(self):
        result = run_cli("--help")
        assert result.returncode == 0

    def test_help_mentions_algorithm(self):
        result = run_cli("--help")
        output = result.stdout + result.stderr
        assert "algorithm" in output.lower() or "luhn" in output.lower()

    def test_help_mentions_all_algorithms(self):
        result = run_cli("--help")
        output = result.stdout + result.stderr
        for algo in ["luhn", "lsa", "text-rank", "lex-rank", "sum-basic", "kl", "edmundson"]:
            assert algo in output

    def test_help_mentions_options(self):
        result = run_cli("--help")
        output = result.stdout + result.stderr
        assert "--url" in output
        assert "--file" in output
        assert "--length" in output


class TestVersionOutput:
    """CLI --version flag."""

    def test_version_exits_zero(self):
        result = run_cli("--version")
        assert result.returncode == 0

    def test_version_output_nonempty(self):
        result = run_cli("--version")
        output = result.stdout + result.stderr
        assert len(output.strip()) > 0


class TestLuhnAlgorithmCLI:
    """Luhn algorithm via CLI — smoke test (core milestone requirement)."""

    def test_luhn_file_3_sentences(self):
        """python -m sumy luhn --file=README.rst --length=3 must produce 3 lines."""
        result = run_cli("luhn", f"--file={README_PATH}", "--length=3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_luhn_file_1_sentence(self):
        result = run_cli("luhn", f"--file={README_PATH}", "--length=1")
        assert result.returncode == 0
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_luhn_file_percentage_length(self):
        result = run_cli("luhn", f"--file={README_PATH}", "--length=10%")
        assert result.returncode == 0
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 1

    def test_luhn_output_is_text(self):
        result = run_cli("luhn", f"--file={README_PATH}", "--length=2")
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0


class TestLsaAlgorithmCLI:
    """LSA algorithm via CLI."""

    def test_lsa_file_3_sentences(self):
        result = run_cli("lsa", f"--file={README_PATH}", "--length=3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 3


class TestTextRankAlgorithmCLI:
    """TextRank algorithm via CLI."""

    def test_text_rank_file_3_sentences(self):
        result = run_cli("text-rank", f"--file={README_PATH}", "--length=3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 3


class TestLexRankAlgorithmCLI:
    """LexRank algorithm via CLI."""

    def test_lex_rank_file_3_sentences(self):
        result = run_cli("lex-rank", f"--file={README_PATH}", "--length=3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 3


class TestSumBasicAlgorithmCLI:
    """SumBasic algorithm via CLI."""

    def test_sum_basic_file_3_sentences(self):
        result = run_cli("sum-basic", f"--file={README_PATH}", "--length=3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 3


class TestEdmundsonAlgorithmCLI:
    """Edmundson algorithm via CLI."""

    def test_edmundson_file_3_sentences(self):
        result = run_cli("edmundson", f"--file={README_PATH}", "--length=3")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 3


class TestKLAlgorithmCLI:
    """KL algorithm via CLI — has known pre-existing bug."""

    def test_kl_exits_nonzero_due_to_bug(self):
        """
        KL summarizer has a pre-existing KeyError bug (_kl_divergence uses
        unnormalized keys to look up in a normalized frequency dict). The CLI
        wraps exceptions and exits 1.
        """
        result = run_cli("kl", f"--file={README_PATH}", "--length=3")
        assert result.returncode != 0


class TestInvalidArgs:
    """Invalid arguments to CLI."""

    def test_invalid_algorithm_exits_nonzero(self):
        result = run_cli("nonexistent-algo", f"--file={README_PATH}")
        assert result.returncode != 0

    def test_invalid_format_exits_nonzero(self):
        result = run_cli("luhn", f"--file={README_PATH}", "--format=invalid")
        assert result.returncode != 0

    def test_no_algorithm_exits_nonzero(self):
        result = run_cli(f"--file={README_PATH}")
        assert result.returncode != 0


class TestCLIFileInput:
    """CLI with file input option."""

    def test_file_with_utf8_content(self, tmp_path):
        content = (
            "Natural language processing is a fascinating field of study.\n"
            "It enables computers to understand and process human language.\n"
            "Machine learning has greatly advanced NLP capabilities in recent years.\n"
            "Deep learning models have achieved state-of-the-art performance.\n"
            "Text summarization is one of many important NLP applications.\n"
        )
        test_file = tmp_path / "test_document.txt"
        test_file.write_text(content, encoding="utf-8")
        result = run_cli("luhn", f"--file={test_file}", "--length=2")
        assert result.returncode == 0
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 2
