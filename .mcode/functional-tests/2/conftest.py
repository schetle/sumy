import collections
import collections.abc
import os

for _n in ("Sequence", "Mapping", "Set", "Iterable"):
    setattr(collections, _n, getattr(collections.abc, _n))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TXT = os.path.join(REPO, "tests", "data", "articles", "svd_converges.txt")
