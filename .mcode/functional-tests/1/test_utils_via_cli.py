import os, subprocess, sys

SUMY = os.path.join(os.path.dirname(sys.executable), "sumy")
ENV = dict(os.environ, NLTK_DATA=os.environ.get("NLTK_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))), ".nltk_data")))
TEXT = "Cats are animals. Dogs are animals too. Birds can fly high. Fish swim in water. Trees grow tall.\n"


def run(args, stdin=None):
    return subprocess.run([SUMY] + args, input=stdin, capture_output=True, text=True, env=ENV, timeout=120)


def test_cli_percentage_items_count_and_stopwords():
    r = run(["lsa", "--length=50%", "--language=english"], TEXT)
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.strip().splitlines()) == 2


def test_cli_integer_items_count():
    r = run(["lsa", "--length=3", "--language=english"], TEXT)
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.strip().splitlines()) == 3


def test_cli_unknown_language_stopwords_error():
    r = run(["lsa", "--language=klingon"], TEXT)
    assert r.returncode != 0
    assert "Stop-words are not available for language klingon" in r.stderr
