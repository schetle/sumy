.. :changelog:

Changelog
=========

0.4.0 (2026-06-08)
------------------
- **Breaking change**: Dropped support for Python 2 and Python < 3.10. Python 3.10+ is now required.
- Deleted ``sumy/_compat.py`` Python 2 compatibility layer.
- Removed all ``from __future__ import`` headers.
- Fixed critical Python 3.10+ bug: ``collections.Sequence`` replaced with ``collections.abc.Sequence``.
- Replaced abandoned ``breadability`` HTML parser with ``readability-lxml``.
- Replaced ``docopt`` CLI parsing with ``click`` 8.x in both ``sumy`` and ``sumy_eval`` entry points.
- Updated NLTK tokenizer to use ``punkt_tab`` resource, fixing CVE-2024-39705 (pickle RCE vulnerability).
- Migrated packaging from ``setup.py``/``setup.cfg``/``MANIFEST.in`` to ``pyproject.toml``.
- Converted test suite to pytest-native style.

0.3.0 (2014-06-07)
------------------
- Added possibility to specify format of input document for URL & stdin. Thanks to `@Lucas-C <https://github.com/Lucas-C>`_.
- Added possibility to specify custom file with stop-words in CLI. Thanks to `@Lucas-C <https://github.com/Lucas-C>`_.
- Added support for French language (added stopwords & stemmer). Thanks to `@Lucas-C <https://github.com/Lucas-C>`_.
- Function ``sumy.utils.get_stop_words`` raises ``LookupError`` instead of ``ValueError`` for unknown language.
- Exception ``LookupError`` is raised for unknown language of stemmer instead of falling silently to ``null_stemmer``.

0.2.1 (2014-01-23)
------------------
- Fixed installation of my own readability fork. Added ``breadability`` to the dependencies instead of it `#8 <https://github.com/miso-belica/sumy/issues/8>`_. Thanks to `@pratikpoddar <https://github.com/pratikpoddar>`_.

0.2.0 (2014-01-18)
------------------
- Removed dependency on SciPy `#7 <https://github.com/miso-belica/sumy/pull/7>`_. Use ``numpy.linalg.svd`` implementation. Thanks to `Shantanu <https://github.com/baali>`_.

0.1.0 (2013-10-20)
------------------
- First public release.
