import os, subprocess, sys, pytest
GO = os.path.join(os.environ["WORKSPACE_DIR"], "sumy", "go")
BIN = os.path.join(GO, "bin", "sumy")
PYROOT = os.path.join(os.environ["WORKSPACE_DIR"], "sumy")
PYSHIM = "/tmp/pysumy.py"

TEXT = ("Dogs are great pets. Dogs love to play and dogs love to run. Cats are also nice pets. "
        "Cats sleep a lot and cats purr. Birds sing songs. Fish swim in water. "
        "Dogs and cats are the most popular pets in the world today.\n")

def run(args, stdin=None):
    return subprocess.run([BIN] + args, input=stdin, capture_output=True, text=True, timeout=30)

def py(args, stdin=None):
    return subprocess.run([sys.executable, PYSHIM] + args, input=stdin, capture_output=True, text=True, timeout=60, cwd=PYROOT)

@pytest.fixture
def txt(tmp_path):
    p = tmp_path / "a.txt"; p.write_text(TEXT); return str(p)

def test_version():
    r = run(["--version"]); assert r.returncode == 0 and r.stdout.strip() == "0.3.0"

def test_help():
    r = run(["--help"]); assert r.returncode == 0 and "--length" in r.stdout

def test_file_count(txt):
    r = run(["luhn", "--length=2", "--file", txt])
    assert r.returncode == 0
    assert r.stdout.splitlines() == ["Dogs love to play and dogs love to run.",
                                     "Dogs and cats are the most popular pets in the world today."]

def test_stdin_no_positional():
    r = run(["--length=2"], stdin=TEXT)
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 2

def test_default_percentage(txt):
    r = run(["--file", txt])  # 20% of 7 sentences -> 1
    assert r.returncode == 0 and len(r.stdout.splitlines()) >= 1

def test_percentage(txt):
    r = run(["--length=50%", "--file", txt])
    assert r.returncode == 0 and 3 <= len(r.stdout.splitlines()) <= 4

def test_document_order(txt):
    out = run(["--length=3", "--file", txt]).stdout.splitlines()
    idx = [TEXT.index(s) for s in out]
    assert idx == sorted(idx)

def test_custom_stopwords(txt, tmp_path):
    sw = tmp_path / "sw.txt"; sw.write_text("dogs\ncats\n")
    r = run(["--length=2", "--stopwords", str(sw), "--file", txt])
    assert r.returncode == 0 and r.stdout.strip()

@pytest.mark.parametrize("lang,text", [
    ("czech", "Pes je dobrý. Pes rád běhá a pes rád skáče. Kočka spí. Kočka a pes jsou přátelé.\n"),
    ("german", "Der Hund ist gut. Der Hund läuft gern und der Hund springt. Die Katze schläft. Hund und Katze sind Freunde.\n"),
    ("french", "Le chien est bon. Le chien aime courir et le chien saute. Le chat dort. Chien et chat sont amis.\n"),
    ("slovak", "Pes je dobrý. Pes rád behá a pes rád skáče. Mačka spí. Mačka a pes sú priatelia.\n"),
])
def test_languages(lang, text):
    r = run(["--length=1", "--language", lang], stdin=text)
    assert r.returncode == 0 and r.stdout.strip()

def test_unknown_language():
    r = run(["--language", "klingon"], stdin=TEXT)
    assert r.returncode == 1 and r.stderr.strip() and not r.stdout

def test_missing_file():
    r = run(["--file", "/nonexistent/x.txt"])
    assert r.returncode == 1 and r.stderr.strip()

def test_invalid_length(txt):
    r = run(["--length=abc", "--file", txt])
    assert r.returncode == 1 and r.stderr.strip()

def test_unknown_flag():
    r = run(["--bogus"]); assert r.returncode != 0

def test_missing_stopwords_file(txt):
    r = run(["--stopwords", "/nonexistent/sw", "--file", txt])
    assert r.returncode == 1

def test_empty_stdin():
    r = run([], stdin=""); assert r.returncode in (0, 1) and r.returncode == 0 or r.stderr

@pytest.mark.skipif(not os.path.exists(PYSHIM), reason="python reference unavailable")
@pytest.mark.parametrize("length", ["1", "2", "3", "30%", "50%"])
def test_parity_with_python(txt, length):
    g = run(["luhn", f"--length={length}", "--file", txt]); p = py(["luhn", f"--length={length}", "--file", txt])
    assert p.returncode == 0, p.stderr
    assert g.stdout.splitlines() == p.stdout.splitlines()
