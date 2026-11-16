import subprocess, sys, os, tempfile
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TEXT = "The cat sat on the mat. The dog chased the cat around the garden. Birds sing in the morning. The mat was red and soft. A dog is a loyal animal."

def run(args):
    return subprocess.run([sys.executable] + args, cwd=REPO, capture_output=True, text=True, timeout=120)

def tmpfile(text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
    return f.name

def py(code):
    return run(["-c", code])

def test_cli_lsa_summary():
    r = run(["-m", "sumy", "lsa", "--length=2", "--file=" + tmpfile(TEXT)])
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.strip().splitlines()) == 2

def test_cli_percentage_length_itemscount():
    r = run(["-m", "sumy", "luhn", "--length=50%", "--file=" + tmpfile("First one here. Second one there. Third one now. Fourth ends.")])
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.strip().splitlines()) <= 2

def test_cli_missing_stopwords_language():
    r = run(["-m", "sumy", "lsa", "--length=1", "--language=klingon", "--file=" + tmpfile("Hello world. Bye.")])
    assert r.returncode != 0

def test_cli_lex_rank_tf_model():
    r = run(["-m", "sumy", "lex-rank", "--length=1", "--file=" + tmpfile(TEXT)])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()

def test_tfmodel_behaviour():
    r = py("""
from sumy.models.tf import TfDocumentModel as M
from sumy.nlp.tokenizers import Tokenizer
m = M(('The','the','Dog','cat'))
assert m.term_frequency('the') == 2 and m.term_frequency('zzz') == 0
assert sorted(m.terms) == ['cat','dog','the']
assert m.most_frequent_terms()[0] == 'the' and len(m.most_frequent_terms(2)) == 2
assert abs(m.magnitude - 6**0.5) < 1e-9
assert m.normalized_term_frequency('the') == 1.0
assert m.normalized_term_frequency('dog', 0.4) == 0.4 + 0.6*0.5
assert repr(m).startswith('<TfDocumentModel')
m2 = M('a b a', Tokenizer('english')); assert m2.term_frequency('a') == 2
for bad in [lambda: M('x'), lambda: M(5), lambda: m.most_frequent_terms(-1)]:
    try: bad()
    except ValueError: pass
    else: raise SystemExit('no ValueError')
print('ok')
""")
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout

def test_utils_behaviour():
    r = py("""
from sumy.utils import *
n=[0]
class A:
    @cached_property
    def v(self):
        n[0]+=1; return 7
a=A(); assert a.v==7 and a.v==7 and n[0]==1
sw = get_stop_words('english'); assert isinstance(sw, frozenset) and 'the' in sw
try: get_stop_words('klingon')
except LookupError: pass
else: raise SystemExit('no LookupError')
assert expand_resource_path('stopwords/english.txt').endswith('english.txt')
assert 'the' in read_stop_words(expand_resource_path('stopwords/english.txt'))
s=[1,2,3,4]
assert ItemsCount(2)(s)==[1,2] and ItemsCount('50%')(s)==[1,2] and ItemsCount('1%')(s)==[1] and ItemsCount('3')(s)==[1,2,3] and ItemsCount(2.0)(s)==[1,2]
assert repr(ItemsCount('5%'))=="<ItemsCount: '5%'>"
print('ok')
""")
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
