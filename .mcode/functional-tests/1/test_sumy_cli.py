import glob, os, subprocess, sys, venv, tempfile
import pytest

VENV = os.path.join(os.environ.get("WORKSPACE_DIR", "/l2l/workspace"), ".venv-sumy", "bin")
REPO = os.path.join(os.environ.get("WORKSPACE_DIR", "/l2l/workspace"), "sumy")
TEXT = ("Automatic summarization is the process of shortening a text document with software. "
        "The goal is to create a summary that retains the most important points of the original document. "
        "Technologies that can make a coherent summary take into account variables such as length, writing style and syntax. "
        "Extractive methods select a subset of existing words, phrases, or sentences from the original text. "
        "Abstractive methods build an internal semantic representation and then use natural language generation to create a summary.\n")


def run(*args, **kw):
    return subprocess.run(list(args), capture_output=True, text=True, timeout=300, **kw)


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    p = tmp_path_factory.mktemp("doc") / "doc.txt"
    p.write_text(TEXT)
    return str(p)


def test_version():
    r = run(f"{VENV}/sumy", "--version")
    assert r.returncode == 0 and "0.3.0" in r.stdout


def test_sumy_eval_runs():
    r = run(f"{VENV}/sumy_eval", "--version")
    assert r.returncode == 0 and "0.3.0" in r.stdout


@pytest.mark.parametrize("algo", ["lsa", "lex-rank", "text-rank", "luhn", "sum-basic", "kl"])
def test_summarizer(algo, doc):
    r = run(f"{VENV}/sumy", algo, "--length=2", f"--file={doc}")
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    assert len(lines) == 2
    for l in lines:
        assert l.strip() in TEXT


def test_missing_file_fails():
    r = run(f"{VENV}/sumy", "lsa", "--file=/nonexistent/x.txt")
    assert r.returncode != 0


def test_invalid_algo_fails():
    r = run(f"{VENV}/sumy", "bogus", "--file=/etc/hostname")
    assert r.returncode != 0


def test_stopwords_installed():
    r = run(f"{VENV}/python", "-I", "-c",
            "from sumy.utils import get_stop_words; s=get_stop_words('english'); assert 'the' in s and len(s)>50; print(len(s))",
            cwd="/tmp")
    assert r.returncode == 0, r.stderr


def test_wheel_install_stopwords_and_cli(tmp_path, doc):
    whl = glob.glob(f"{REPO}/dist/sumy-0.3.0-*.whl")
    assert whl
    v = str(tmp_path / "v")
    venv.create(v, with_pip=True)
    r = run(f"{v}/bin/pip", "install", "-q", whl[0], "numpy")
    assert r.returncode == 0, r.stderr
    r = run(f"{v}/bin/python", "-I", "-c",
            "import sumy,os;from sumy.utils import get_stop_words;print(len(get_stop_words('english')));"
            "print(os.path.exists(os.path.join(os.path.dirname(sumy.__file__),'py.typed')))", cwd="/tmp")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[-1] == "True"
    r = run(f"{v}/bin/sumy", "--version")
    assert "0.3.0" in r.stdout
    r = run(f"{v}/bin/sumy", "lex-rank", "--length=2", f"--file={doc}", env={**os.environ, "PATH": f"{v}/bin:" + os.environ["PATH"]})
    assert r.returncode == 0, r.stderr
    assert len([l for l in r.stdout.splitlines() if l.strip()]) == 2
