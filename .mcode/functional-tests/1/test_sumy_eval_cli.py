# -*- coding: utf8 -*-
"""Functional tests for the `sumy_eval` evaluation console script: real
end-to-end comparison of a generated summary against a reference summary,
exercising ROUGE / content-based / co-selection metrics.
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


EXPECTED_METRICS = [
    "Precision", "Recall", "F-score",
    "Cosine similarity", "Cosine similarity (document)",
    "Unit overlap", "Unit overlap (document)",
    "Rouge-1", "Rouge-2", "Rouge-L (Sentence Level)", "Rouge-L (Summary Level)",
]


class TestSumyEvalHappyPath:
    def test_eval_via_file(self, sumy_eval_bin, sample_en_file, reference_summary_file):
        result = run(
            sumy_eval_bin, "luhn", str(reference_summary_file),
            "--length=2", "--language=english",
            f"--file={sample_en_file}", "--format=plaintext",
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        for metric in EXPECTED_METRICS:
            assert metric in result.stdout, f"missing metric {metric!r} in output: {result.stdout}"

        # Precision/Recall/F-score should be real floats in [0, 1].
        for line in result.stdout.strip().splitlines():
            name, _, value = line.partition(":")
            if name.strip() in ("Precision", "Recall", "F-score", "Rouge-1"):
                score = float(value.strip())
                assert 0.0 <= score <= 1.0

    def test_eval_via_stdin(self, sumy_eval_bin, sample_en_file, reference_summary_file):
        input_text = sample_en_file.read_text(encoding="utf-8")
        result = run(
            sumy_eval_bin, "lex-rank", str(reference_summary_file),
            "--length=2", "--language=english",
            input_text=input_text,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        for metric in EXPECTED_METRICS:
            assert metric in result.stdout

    def test_eval_exact_reference_match_gives_perfect_scores(self, sumy_eval_bin, sample_en_file, reference_summary_file):
        """The reference_summary fixture is exactly the top-2 'luhn' sentences
        of sample_en_file, so evaluating luhn against it should score a
        perfect match on precision/recall/rouge-1."""
        result = run(
            sumy_eval_bin, "luhn", str(reference_summary_file),
            "--length=2", "--language=english",
            f"--file={sample_en_file}", "--format=plaintext",
        )
        assert result.returncode == 0
        values = dict(
            (line.split(":")[0].strip(), float(line.split(":")[1].strip()))
            for line in result.stdout.strip().splitlines()
        )
        assert values["Precision"] == 1.0
        assert values["Recall"] == 1.0
        assert values["Rouge-1"] == 1.0
