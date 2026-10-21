# -*- coding: utf8 -*-
"""HAPPY_PATH functional tests: run every summarization algorithm exposed by
the `sumy` CLI against real plaintext input, via both --file and stdin, and
confirm each produces a real, sensible multi-sentence extractive summary.

This also regression-tests the bug found & fixed during this milestone:
sumy/__main__.py opens --file in binary mode, and PlaintextParser used to do
str(bytes_obj) on it (producing a "b'...'" bytes-repr) instead of decoding.
If that regressed, output would contain "b'" and/or backslash-escapes
instead of real text.
"""

import subprocess

import pytest

ALGORITHMS = ["luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"]


def run(binary, *args, input_text=None):
    return subprocess.run(
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=60,
        input=input_text,
    )


def assert_sensible_summary(output_text, forbidden_source_text=None):
    """A sensible extractive summary: non-empty, multiple lines, real
    decoded text (no Python bytes-repr artifacts)."""
    assert output_text.strip(), "summary output was empty"
    lines = [line for line in output_text.strip().splitlines() if line.strip()]
    assert len(lines) >= 2, f"expected a multi-sentence summary, got: {output_text!r}"
    assert "b'" not in output_text and 'b"' not in output_text, (
        f"output looks like a bytes repr, not decoded text: {output_text!r}"
    )
    # Each extracted line should be a real sentence (reasonably long, has spaces).
    for line in lines:
        assert " " in line, f"line does not look like a real sentence: {line!r}"


class TestAlgorithmsViaFile:
    """Run each algorithm with --file=<path>, --format=plaintext."""

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_algorithm_produces_sensible_summary_via_file(self, sumy_bin, sample_en_file, algorithm):
        result = run(
            sumy_bin,
            algorithm,
            "--length=2",
            "--language=english",
            "--format=plaintext",
            f"--file={sample_en_file}",
        )
        assert result.returncode == 0, (
            f"`sumy {algorithm} --file=...` crashed/exited non-zero.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert_sensible_summary(result.stdout)


class TestAlgorithmsViaStdin:
    """Run each algorithm with input piped via stdin, --format=plaintext."""

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_algorithm_produces_sensible_summary_via_stdin(self, sumy_bin, sample_en_file, algorithm):
        input_text = sample_en_file.read_text(encoding="utf-8")
        result = run(
            sumy_bin,
            algorithm,
            "--length=2",
            "--language=english",
            "--format=plaintext",
            input_text=input_text,
        )
        assert result.returncode == 0, (
            f"`sumy {algorithm}` (stdin) crashed/exited non-zero.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert_sensible_summary(result.stdout)


class TestFileAndStdinProduceEquivalentOutput:
    """--file and stdin should take the same decoded text through the same
    pipeline, so for a given algorithm they should produce the same
    extractive summary from the same input."""

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_file_and_stdin_agree(self, sumy_bin, sample_en_file, algorithm):
        input_text = sample_en_file.read_text(encoding="utf-8")

        file_result = run(
            sumy_bin, algorithm, "--length=2", "--language=english",
            "--format=plaintext", f"--file={sample_en_file}",
        )
        stdin_result = run(
            sumy_bin, algorithm, "--length=2", "--language=english",
            "--format=plaintext", input_text=input_text,
        )

        assert file_result.returncode == stdin_result.returncode
        if file_result.returncode == 0:
            assert file_result.stdout == stdin_result.stdout
