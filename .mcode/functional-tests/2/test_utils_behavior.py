import os, subprocess, sys
import pytest
import sumy
from sumy import utils
from sumy.utils import expand_resource_path, get_stop_words, read_stop_words


def test_get_stop_words_english():
    w = get_stop_words("english")
    assert isinstance(w, frozenset)
    assert "the" in w and len(w) > 50


def test_get_stop_words_unknown_language():
    with pytest.raises(LookupError):
        get_stop_words("klingon_nonexistent")


def test_expand_resource_path():
    p = expand_resource_path("stopwords/english.txt")
    assert os.path.isabs(p)
    assert p.endswith(os.path.join("data", "stopwords", "english.txt"))
    assert os.path.exists(p)


def test_read_stop_words_file(tmp_path):
    f = tmp_path / "sw.txt"
    f.write_bytes(b"alpha\nbeta  \ngamma\n")
    assert read_stop_words(str(f)) == frozenset(["alpha", "beta", "gamma"])


def test_read_stop_words_missing_file(tmp_path):
    with pytest.raises(IOError):
        read_stop_words(str(tmp_path / "nope.txt"))


@pytest.mark.parametrize("fn", [expand_resource_path, get_stop_words, read_stop_words])
def test_docstrings_present(fn):
    assert fn.__doc__ and fn.__doc__.strip()
    assert ":param" in fn.__doc__ and ":returns:" in fn.__doc__


def test_docstring_raises_documented():
    assert "LookupError" in get_stop_words.__doc__
    assert "IOError" in read_stop_words.__doc__
    assert utils.__doc__


def test_docstrings_via_subprocess():
    out = subprocess.check_output([sys.executable, "-c",
        "from sumy.utils import get_stop_words as g; print(g.__doc__)"])
    assert b"stop-words" in out
