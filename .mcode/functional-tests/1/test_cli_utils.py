import os, subprocess, sys
import pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PY = os.path.join(ROOT, ".venv", "bin", "python")
SUMY = os.path.join(ROOT, ".venv", "bin", "sumy")
TEXT = ("Cats are small animals. Cats like to sleep all day long. Dogs are loyal animals. "
        "Dogs like to play fetch in the park. Birds can fly high in the sky. Fish swim in water.")

def run(args, stdin=None):
    return subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=120, cwd=ROOT)

def test_utils_imports_and_docstring():
    r = run([PY, "-c", "import sumy.utils as u; print(bool(u.__doc__))"])
    assert r.returncode == 0 and "True" in r.stdout, r.stderr

def test_get_stop_words():
    r = run([PY, "-c", "from sumy.utils import get_stop_words; s=get_stop_words('english'); print(len(s), 'the' in s)"])
    assert r.returncode == 0, r.stderr
    assert "True" in r.stdout

def test_get_stop_words_unknown_language():
    r = run([PY, "-c", "from sumy.utils import get_stop_words; get_stop_words('klingon')"])
    assert r.returncode != 0

def test_cli_version():
    r = run([SUMY, "--version"])
    assert r.returncode == 0 and r.stdout.strip()

def test_cli_summarize_stdin():
    r = run([SUMY, "lsa", "--length=2", "--format=plaintext", "--language=english", "--file=/dev/stdin"], stdin=TEXT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()
