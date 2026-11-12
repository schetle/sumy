import os, subprocess, sys
import pytest

from conftest import REPO, TXT
SHIM = ("import collections, collections.abc, sys\n"
        "[setattr(collections, n, getattr(collections.abc, n)) for n in ('Sequence','Mapping','Set','Iterable')]\n"
        "from sumy.__main__ import main\n"
        "sys.exit(main(sys.argv[1:]))\n")
METHODS = ["luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"]


def run(args, stdin=None):
    return subprocess.run([sys.executable, "-c", SHIM] + args, cwd=REPO, input=stdin,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("method", METHODS)
def test_cli_method_file(method):
    args = [method, "--length=2", "--file=" + TXT, "--format=plaintext"]
    if method == "edmundson":
        pytest.skip("edmundson CLI needs bonus/stigma words not provided by CLI")
    r = run(args)
    assert r.returncode == 0, r.stderr
    assert 1 <= len([l for l in r.stdout.splitlines() if l.strip()]) <= 2


@pytest.mark.parametrize("method", [m for m in METHODS if m != "edmundson"])
def test_cli_method_stdin(method):
    text = open(TXT, encoding="utf8").read()
    r = run([method, "--length=3", "--format=plaintext"], stdin=text)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()


def test_cli_version():
    r = run(["--version"])
    assert r.returncode == 0 and r.stdout.strip()


def test_cli_invalid_method():
    r = run(["bogus"])
    assert r.returncode != 0


def test_cli_missing_file():
    r = run(["lsa", "--file=/nonexistent/x.txt"])
    assert r.returncode != 0
