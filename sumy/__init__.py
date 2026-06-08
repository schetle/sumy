# -*- coding: utf8 -*-

from importlib.metadata import version, PackageNotFoundError

__author__ = "Michal Belica"

try:
    __version__ = version("sumy")
except PackageNotFoundError:
    __version__ = "0.4.0-dev"
