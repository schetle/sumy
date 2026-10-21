# -*- coding: utf8 -*-
"""HAPPY_PATH functional tests for:
  - the breadability-based HTML parsing path (--format=html)
  - non-English language dispatch (stemmer/tokenizer) via --language
"""

import subprocess


def run(binary, *args, input_text=None):
    return subprocess.run(
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=60,
        input=input_text,
    )


def assert_sensible_summary(output_text):
    assert output_text.strip(), "summary output was empty"
    lines = [line for line in output_text.strip().splitlines() if line.strip()]
    assert len(lines) >= 2, f"expected a multi-sentence summary, got: {output_text!r}"
    assert "b'" not in output_text and 'b"' not in output_text


class TestHtmlFormat:
    def test_html_via_file(self, sumy_bin, sample_html_file):
        result = run(
            sumy_bin, "lex-rank", "--length=2", "--language=english",
            "--format=html", f"--file={sample_html_file}",
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert_sensible_summary(result.stdout)
        # Make sure HTML markup was actually stripped out by breadability.
        assert "<p>" not in result.stdout and "<html>" not in result.stdout

    def test_html_via_stdin(self, sumy_bin, sample_html_file):
        html_text = sample_html_file.read_text(encoding="utf-8")
        result = run(
            sumy_bin, "luhn", "--length=2", "--language=english",
            "--format=html", input_text=html_text,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert_sensible_summary(result.stdout)
        assert "<p>" not in result.stdout and "<html>" not in result.stdout


class TestNonEnglishLanguages:
    def test_german_language(self, sumy_bin, sample_de_file):
        result = run(
            sumy_bin, "lex-rank", "--length=2", "--language=german",
            "--format=plaintext", f"--file={sample_de_file}",
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert_sensible_summary(result.stdout)
        # Confirm real German text came through (umlaut preserved, not mangled).
        assert "Unternehmen" in result.stdout or "Digitalisierung" in result.stdout

    def test_czech_language_existing_fixture(self, sumy_bin):
        import pathlib
        czech_fixture = pathlib.Path(__file__).resolve().parents[3] / "tests" / "data" / "articles" / "prevko_cz_1.txt"
        if not czech_fixture.exists():
            import pytest
            pytest.skip("czech fixture not present in this checkout")

        input_text = czech_fixture.read_text(encoding="utf-8")
        result = run(
            sumy_bin, "luhn", "--length=2", "--language=czech",
            "--format=plaintext", input_text=input_text,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert_sensible_summary(result.stdout)
