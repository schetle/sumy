import os, subprocess, sys
import docutils.core, docutils.utils

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TEXT = "Natural language processing is a field of computer science. It studies how computers understand human language. Summarization reduces a document to its most important sentences. Many algorithms exist, such as LSA, LexRank and TextRank."

def run(args, stdin=None):
    return subprocess.run(["sumy"] + args, input=stdin, capture_output=True, text=True, timeout=120)

def test_readme_renders_without_warnings():
    src = open(os.path.join(REPO, "README.rst"), encoding="utf-8").read()
    assert src.startswith(".. This is a Python 3.12 modernization of sumy.\n\n.. _Python:")
    _, pub = docutils.core.publish_programmatically(
        source_class=docutils.io.StringInput, source=src, source_path=None,
        destination_class=docutils.io.StringOutput, destination=None, destination_path=None,
        reader=None, reader_name="standalone", parser=None, parser_name="restructuredtext",
        writer=None, writer_name="html", settings=None, settings_spec=None,
        settings_overrides={"report_level": 2, "halt_level": 2}, config_section=None, enable_exit_status=False)
    html = pub.writer.output
    import re
    visible = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    assert "modernization of sumy" not in visible
    assert "<h1" in visible or "<h2" in visible

def test_pyproject_readme_metadata():
    from importlib.metadata import metadata
    desc = metadata("sumy").get("Description") or ""
    assert "sumy" in desc.lower()

def test_cli_version():
    r = run(["--version"])
    assert r.returncode == 0, r.stderr

def test_cli_help():
    r = run(["--help"])
    assert r.returncode == 0 and "Usage" in r.stdout + r.stderr or "sumy" in r.stdout

def test_cli_lsa_summarize():
    r = run(["lsa", "--length=2", "--format=plaintext"], stdin=TEXT)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()

def test_cli_invalid_args():
    r = run(["nonexistent-summarizer"])
    assert r.returncode != 0
