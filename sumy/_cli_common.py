# -*- coding: utf8 -*-

from . import __version__
from .parsers.html import HtmlParser
from .parsers.plaintext import PlaintextParser

HEADERS = {
    "User-Agent": "Sumy (Automatic text summarizer) Version/%s" % __version__,
}
PARSERS = {
    "html": HtmlParser,
    "plaintext": PlaintextParser,
}
