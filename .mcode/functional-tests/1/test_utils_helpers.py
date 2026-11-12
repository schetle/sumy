# -*- coding: utf8 -*-
import os
import pytest
import sumy.utils as u
from sumy.utils import cached_property, expand_resource_path, get_stop_words, read_stop_words, ItemsCount


def test_module_imports_with_docstring():
    assert u.__doc__ and "Utility helpers" in u.__doc__
    for n in ("cached_property", "expand_resource_path", "get_stop_words", "read_stop_words", "ItemsCount"):
        assert hasattr(u, n)


def test_cached_property_memoizes():
    class A(object):
        calls = 0

        @cached_property
        def v(self):
            A.calls += 1
            return 42
    a = A()
    assert a.v == 42 and a.v == 42
    assert A.calls == 1


def test_expand_resource_path():
    p = expand_resource_path("stopwords/english.txt")
    assert os.path.isabs(p) and os.path.exists(p)
    assert p.endswith(os.path.join("data", "stopwords", "english.txt"))


def test_get_stop_words_english():
    w = get_stop_words("english")
    assert isinstance(w, frozenset) and "the" in w


def test_get_stop_words_unknown_language():
    with pytest.raises(LookupError):
        get_stop_words("klingon")


def test_read_stop_words_file(tmp_path):
    f = tmp_path / "sw.txt"
    f.write_bytes("foo\nbar \nžluťoučký\n".encode("utf8"))
    assert read_stop_words(str(f)) == frozenset(["foo", "bar", u"žluťoučký"])


def test_items_count_variants():
    s = list(range(10))
    assert ItemsCount(3)(s) == [0, 1, 2]
    assert ItemsCount("4")(s) == [0, 1, 2, 3]
    assert ItemsCount("50%")(s) == s[:5]
    assert ItemsCount("1%")(s) == s[:1]
    assert ItemsCount(2.7)(s) == [0, 1]
    assert "ItemsCount" in repr(ItemsCount(3))


def test_items_count_invalid_string():
    with pytest.raises(ValueError):
        ItemsCount("abc")([1, 2])
