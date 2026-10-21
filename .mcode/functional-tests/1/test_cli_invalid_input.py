# -*- coding: utf8 -*-
"""Negative/edge-case functional tests for the `sumy` CLI: invalid format,
missing file, invalid options.
"""

import subprocess


def run(binary, *args, input_text=None):
    return subprocess.run(
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=30,
        input=input_text,
    )


class TestInvalidFormat:
    def test_unsupported_format_value(self, sumy_bin):
        result = run(sumy_bin, "luhn", "--format=badformat", input_text="Some text. More text.\n")
        assert result.returncode != 0
        assert "Unsupported format" in (result.stdout + result.stderr)


class TestNotFound:
    def test_nonexistent_file(self, sumy_bin):
        result = run(sumy_bin, "luhn", "--file=/tmp/this_file_does_not_exist_ft_sumy_12345.txt")
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "No such file" in combined or "FileNotFoundError" in combined


class TestInvalidOptions:
    def test_both_url_and_file_rejected(self, sumy_bin):
        # docopt usage patterns don't allow --url and --file together.
        result = run(sumy_bin, "luhn", "--url=http://example.com", "--file=/tmp/x.txt")
        assert result.returncode != 0

    def test_missing_algorithm_argument(self, sumy_bin):
        result = run(sumy_bin, "--length=2")
        assert result.returncode != 0
