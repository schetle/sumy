import os, subprocess, sys, tempfile
import pytest
import sumy.utils as u
from sumy.utils import cached_property, expand_resource_path, get_stop_words, read_stop_words, ItemsCount


def test_module_docstring_and_future_import():
    assert u.__doc__ and u.__doc__.strip()
    r = subprocess.run([sys.executable, "-c", "import sumy.utils as u; print(bool(u.__doc__))"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "True" in r.stdout


def test_cached_property_memoizes():
    class A(object):
        n = 0
        @cached_property
        def v(self):
            A.n += 1
            return 42
    a = A()
    assert a.v == 42 and a.v == 42 and A.n == 1


def test_expand_resource_path():
    p = expand_resource_path("stopwords/english.txt")
    assert os.path.isfile(p) and p.endswith(os.path.join("data", "stopwords", "english.txt"))


def test_get_stop_words_english():
    w = get_stop_words("english")
    assert "the" in w and isinstance(w, frozenset)


def test_get_stop_words_unknown_language():
    with pytest.raises(LookupError):
        get_stop_words("klingon")


def test_read_stop_words_file():
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(b"foo\nbar \nbaz\n")
    try:
        assert read_stop_words(f.name) == frozenset(["foo", "bar", "baz"])
    finally:
        os.unlink(f.name)


def test_read_stop_words_missing_file():
    with pytest.raises(IOError):
        read_stop_words("/nonexistent/file.txt")


@pytest.mark.parametrize("val,expected", [
    (3, [0, 1, 2]), ("2", [0, 1]), (2.9, [0, 1]), ("50%", [0, 1, 2, 3, 4]),
    ("1%", [0]), ("100%", list(range(10))), (0, []),
])
def test_items_count(val, expected):
    assert list(ItemsCount(val)(list(range(10)))) == expected


def test_items_count_repr_and_invalid():
    assert "ItemsCount" in repr(ItemsCount(5))
    with pytest.raises(ValueError):
        ItemsCount("abc")([1, 2])


def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "sumy", "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 or "usage" in (r.stdout + r.stderr).lower()
