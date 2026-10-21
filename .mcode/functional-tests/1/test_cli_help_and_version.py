# -*- coding: utf8 -*-
"""HELP_OUTPUT / VERSION_OUTPUT functional tests for the `sumy` and
`sumy_eval` console scripts.
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


class TestSumyVersionAndHelp:
    def test_version_output(self, sumy_bin):
        result = run(sumy_bin, "--version")
        assert result.returncode == 0
        assert result.stdout.strip() == "0.3.0"

    def test_help_output(self, sumy_bin):
        result = run(sumy_bin, "--help")
        assert result.returncode == 0
        assert "Usage" in result.stdout
        assert "lex-rank" in result.stdout
        assert "--file" in result.stdout

    def test_no_arguments_prints_usage_and_fails(self, sumy_bin):
        # docopt-ng should still require a command/flag to be given.
        result = run(sumy_bin, "--length=2")
        assert result.returncode != 0


class TestSumyEvalVersionAndHelp:
    def test_version_output(self, sumy_eval_bin):
        result = run(sumy_eval_bin, "--version")
        assert result.returncode == 0
        assert result.stdout.strip() == "0.3.0"

    def test_help_output(self, sumy_eval_bin):
        result = run(sumy_eval_bin, "--help")
        assert result.returncode == 0
        assert "Usage" in result.stdout
        assert "reference_summary" in result.stdout
