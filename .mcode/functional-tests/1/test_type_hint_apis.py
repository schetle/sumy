import os, subprocess, json, pathlib
import pytest

PY = os.path.join(os.environ.get("WORKSPACE_DIR", "/l2l/workspace"), ".venv-sumy/bin/python")
REPO = str(pathlib.Path(__file__).resolve().parents[3])


def run(code, args=()):
    r = subprocess.run([PY, "-c", code, *args], capture_output=True, text=True, cwd=REPO, timeout=120)
    return r


def val(code):
    r = run("import json\n" + code)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_cached_property_memoizes():
    assert val("""
from sumy.utils import cached_property
class A:
    n = 0
    @cached_property
    def x(self):
        A.n += 1
        return 5
a = A(); a.x; a.x
print(json.dumps([a.x, A.n]))""") == [5, 1]


def test_expand_resource_path():
    p = val("from sumy.utils import expand_resource_path as e\nprint(json.dumps(e('stopwords/english.txt')))")
    assert p.endswith(os.path.join("sumy", "data", "stopwords", "english.txt")) and os.path.exists(p)


def test_get_stop_words_english():
    r = val("from sumy.utils import get_stop_words as g\ns=g('english')\nprint(json.dumps([type(s).__name__, 'the' in s]))")
    assert r == ["frozenset", True]


def test_get_stop_words_unknown_raises():
    assert val("""
from sumy.utils import get_stop_words as g
try:
    g('klingon'); print(json.dumps('no'))
except LookupError: print(json.dumps('lookup'))""") == "lookup"


def test_read_stop_words(tmp_path):
    f = tmp_path / "w.txt"; f.write_bytes(b"a\nb\n")
    r = run("from sumy.utils import read_stop_words as r;import sys;print(sorted(r(sys.argv[1])))", [str(f)])
    assert r.returncode == 0 and r.stdout.strip() == "['a', 'b']"


def test_items_count_variants():
    assert val("""
from sumy.utils import ItemsCount as I
s=[1,2,3,4,5,6,7,8,9,10]
print(json.dumps([list(I(3)(s)), list(I('4')(s)), list(I('50%')(s)), list(I('1%')(s)), list(I(2.9)(s))]))""") == [[1,2,3],[1,2,3,4],[1,2,3,4,5],[1],[1,2]]


def test_items_count_unsupported_returns_none():
    assert val("""
from sumy.utils import ItemsCount as I
print(json.dumps(I(None)([1,2])))""") is None


def test_items_count_repr():
    r = run("from sumy.utils import ItemsCount as I;print(repr(I(3)))")
    assert r.stdout.strip() == "<ItemsCount: 3>"


def test_null_stemmer():
    r = run("from sumy.nlp.stemmers import null_stemmer as n;print(n('HeLLo'))")
    assert r.stdout.strip() == "hello"


def test_stemmer_english():
    r = run("from sumy.nlp.stemmers import Stemmer;print(Stemmer('english')('running'))")
    assert r.returncode == 0 and r.stdout.strip() == "run"


def test_stemmer_czech_and_bytes():
    r = run("from sumy.nlp.stemmers import Stemmer;from sumy.nlp.stemmers.czech import stem_word;print(Stemmer('czech')('knihami'), stem_word('knihami'), stem_word('knihami'.encode()), stem_word('knihami', True))")
    assert r.returncode == 0
    a = r.stdout.split()
    assert a[0] == a[1] == a[2] and a[0].startswith("knih")


def test_stemmer_unknown_language():
    r = run("""
from sumy.nlp.stemmers import Stemmer
try: Stemmer('klingon')
except LookupError: print('lookup')""")
    assert r.stdout.strip() == "lookup"


def test_sumy_cli_summarize(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("Cats are small animals. Dogs are loyal pets. Birds can fly high. Fish live in water. Horses run fast.")
    r = subprocess.run([PY, "-m", "sumy.__main__", "lex-rank", "--length=2", "--language=english", f"--file={f}"],
                       capture_output=True, text=True, cwd=REPO, timeout=120)
    assert r.returncode == 0, r.stderr
    assert len([l for l in r.stdout.splitlines() if l.strip()]) == 2
