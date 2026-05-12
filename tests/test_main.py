import pytest

from io import StringIO

from sumy.__main__ import build_parser, handle_arguments, __version__


class TestMain:
    def test_ok_args(self):
        parser = build_parser()
        args = parser.parse_args("luhn --url=URL --format=html".split())
        assert args.method == "luhn"
        assert args.url == "URL"
        assert args.format == "html"

    def test_args_none(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_args_just_command(self):
        parser = build_parser()
        args = parser.parse_args(["lsa"])
        assert args.method == "lsa"
        assert args.url is None
        assert args.file is None
        assert args.format is None
        assert args.language == "english"
        assert args.length == "20%"
        assert args.stopwords is None

    def test_args_invalid_command(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["invalid_method"])

    def test_args_url_and_file(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args("lsa --url=URL --file=FILE".split())

    def test_handle_default_arguments(self):
        parser = build_parser()
        args = parser.parse_args(["lsa"])
        handle_arguments(args, default_input_stream=StringIO("Whatever."))

    def test_handle_wrong_format(self):
        parser = build_parser()
        args = parser.parse_args(["lsa", "--url=URL", "--format=plaintext"])
        # plaintext is valid, but let's test an actually invalid format
        # argparse restricts choices, so we need to set it manually
        args.format = "text"
        args.url = "URL"
        with pytest.raises(ValueError):
            handle_arguments(args, default_input_stream=StringIO("Whatever."))

    def test_all_methods_accepted(self):
        parser = build_parser()
        for method in ["luhn", "edmundson", "lsa", "text-rank", "lex-rank", "sum-basic", "kl"]:
            args = parser.parse_args([method])
            assert args.method == method

    def test_version(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_length_argument(self):
        parser = build_parser()
        args = parser.parse_args(["luhn", "--length", "5"])
        assert args.length == "5"

    def test_language_argument(self):
        parser = build_parser()
        args = parser.parse_args(["luhn", "--language", "czech"])
        assert args.language == "czech"
