# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import inspect

import sumy.utils


def _public_functions():
    for name, function in inspect.getmembers(sumy.utils, inspect.isfunction):
        if not name.startswith("_") and function.__module__ == sumy.utils.__name__:
            yield name, function


def test_public_functions_are_found():
    names = [name for name, _ in _public_functions()]

    assert "get_stop_words" in names


def test_public_functions_have_docstring():
    missing = [
        name for name, function in _public_functions()
        if not (function.__doc__ and function.__doc__.strip())
    ]

    assert missing == []
