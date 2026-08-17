"""
Functional tests for sumy_eval CLI evaluation metrics.
"""
import os
import subprocess
import tempfile
import pytest

SUMY_EVAL = "/tmp/sumy-venv/bin/sumy_eval"
TEST_ARTICLE = "/l2l/workspace/sumy/tests/data/articles/svd_converges.txt"


def run_sumy_eval(*args, input_text=None):
    result = subprocess.run(
        [SUMY_EVAL, *args],
        capture_output=True,
        text=True,
        timeout=60,
        input=input_text,
    )
    return result


@pytest.fixture(scope="module")
def reference_summary_file():
    """Create a temporary reference summary file for use in eval tests."""
    content = (
        "The system processes data and returns evaluation metrics. "
        "The method uses advanced algorithms for text analysis and summarization."
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        fname = f.name
    yield fname
    os.unlink(fname)


class TestSumyEvalLsa:
    """sumy_eval lsa with a reference summary file produces metrics"""

    def test_eval_lsa_exits_zero(self, reference_summary_file):
        result = run_sumy_eval(
            "lsa", reference_summary_file,
            "--file", TEST_ARTICLE,
            "--format", "plaintext",
            "--length", "2",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_eval_lsa_shows_precision(self, reference_summary_file):
        result = run_sumy_eval(
            "lsa", reference_summary_file,
            "--file", TEST_ARTICLE,
            "--format", "plaintext",
            "--length", "2",
        )
        assert result.returncode == 0
        assert "Precision" in result.stdout

    def test_eval_lsa_shows_recall(self, reference_summary_file):
        result = run_sumy_eval(
            "lsa", reference_summary_file,
            "--file", TEST_ARTICLE,
            "--format", "plaintext",
            "--length", "2",
        )
        assert result.returncode == 0
        assert "Recall" in result.stdout

    def test_eval_lsa_shows_rouge_1(self, reference_summary_file):
        result = run_sumy_eval(
            "lsa", reference_summary_file,
            "--file", TEST_ARTICLE,
            "--format", "plaintext",
            "--length", "2",
        )
        assert result.returncode == 0
        assert "Rouge-1" in result.stdout

    def test_eval_lsa_produces_numeric_results(self, reference_summary_file):
        result = run_sumy_eval(
            "lsa", reference_summary_file,
            "--file", TEST_ARTICLE,
            "--format", "plaintext",
            "--length", "2",
        )
        assert result.returncode == 0
        # Check at least one metric has a float value
        lines = result.stdout.strip().split("\n")
        metric_lines = [l for l in lines if ":" in l]
        assert len(metric_lines) > 0
        for line in metric_lines:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    float(parts[1].strip())
                    return  # Found at least one numeric metric
                except ValueError:
                    pass
        pytest.fail("No numeric metrics found in output")


class TestSumyEvalLuhn:
    """sumy_eval luhn method"""

    def test_eval_luhn_exits_zero(self, reference_summary_file):
        result = run_sumy_eval(
            "luhn", reference_summary_file,
            "--file", TEST_ARTICLE,
            "--format", "plaintext",
            "--length", "2",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_eval_luhn_shows_metrics(self, reference_summary_file):
        result = run_sumy_eval(
            "luhn", reference_summary_file,
            "--file", TEST_ARTICLE,
            "--format", "plaintext",
            "--length", "2",
        )
        assert result.returncode == 0
        assert "Precision" in result.stdout
        assert "Recall" in result.stdout


class TestSumyEvalInvalidArgs:
    """sumy_eval invalid arguments"""

    def test_eval_missing_reference_exits_nonzero(self):
        result = run_sumy_eval("lsa", "/nonexistent/reference.txt",
                               "--file", TEST_ARTICLE, "--format", "plaintext")
        assert result.returncode != 0

    def test_eval_invalid_method_exits_nonzero(self, reference_summary_file):
        result = run_sumy_eval("invalid_method", reference_summary_file,
                               "--file", TEST_ARTICLE, "--format", "plaintext")
        assert result.returncode != 0
