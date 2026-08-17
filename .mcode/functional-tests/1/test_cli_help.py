"""
Functional tests for sumy CLI help and version output.
"""
import subprocess
import pytest

SUMY = "/tmp/sumy-venv/bin/sumy"
SUMY_EVAL = "/tmp/sumy-venv/bin/sumy_eval"


def run_cli(*args, input_text=None, executable=SUMY):
    result = subprocess.run(
        [executable, *args],
        capture_output=True,
        text=True,
        timeout=30,
        input=input_text,
    )
    return result


class TestSumyHelp:
    """sumy --help output"""

    def test_sumy_help_exits_zero(self):
        result = run_cli("--help")
        assert result.returncode == 0

    def test_sumy_help_shows_usage(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "Usage" in result.stdout or "usage" in result.stdout.lower()

    def test_sumy_help_shows_methods(self):
        result = run_cli("--help")
        assert "lsa" in result.stdout
        assert "luhn" in result.stdout

    def test_sumy_help_shows_options(self):
        result = run_cli("--help")
        assert "--length" in result.stdout
        assert "--language" in result.stdout


class TestSumyVersion:
    """sumy --version output"""

    def test_sumy_version_exits_zero(self):
        result = run_cli("--version")
        assert result.returncode == 0

    def test_sumy_version_shows_version_string(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0


class TestSumyEvalHelp:
    """sumy_eval --help output"""

    def test_sumy_eval_help_exits_zero(self):
        result = run_cli("--help", executable=SUMY_EVAL)
        assert result.returncode == 0

    def test_sumy_eval_help_shows_usage(self):
        result = run_cli("--help", executable=SUMY_EVAL)
        assert result.returncode == 0
        assert "Usage" in result.stdout or "usage" in result.stdout.lower()

    def test_sumy_eval_help_shows_methods(self):
        result = run_cli("--help", executable=SUMY_EVAL)
        assert "lsa" in result.stdout or "luhn" in result.stdout

    def test_sumy_eval_version_exits_zero(self):
        result = run_cli("--version", executable=SUMY_EVAL)
        assert result.returncode == 0
