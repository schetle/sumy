import io
import os
import subprocess
import sys

import docutils.core
import docutils.utils

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def _run(*args):
    return subprocess.run([sys.executable, "-m", "sumy", *args],
                          capture_output=True, text=True, timeout=120, cwd=ROOT)

def _render():
    src = open(os.path.join(ROOT, "README.rst"), encoding="utf-8").read()
    err = io.StringIO()
    html = docutils.core.publish_string(src, writer="html",
        settings_overrides={"warning_stream": err, "report_level": 2}).decode()
    return src, html, err.getvalue()

def test_readme_starts_with_comment():
    src, _, _ = _render()
    assert src.startswith(".. This is a Python 3.12 modernization of sumy.\n\n.. _Python:")

def test_readme_no_warnings():
    _, _, err = _render()
    assert err.strip() == "", err

def test_comment_not_rendered():
    _, html, _ = _render()
    import re
    visible = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    assert "modernization of sumy" not in visible

def test_cli_help():
    r = _run("--help")
    assert r.returncode == 0 and "sumy" in (r.stdout + r.stderr).lower()

def test_cli_lsa_summarizes_text(tmp_path):
    text = ("Cats are small animals. Cats like to sleep all day. Dogs are loyal animals. "
            "Dogs like to run outside. Birds can fly in the sky. Fish swim in water. "
            "Many people keep pets at home.")
    p = tmp_path / "in.txt"
    p.write_text(text)
    r = _run("lsa", "--length=2", "--file=" + str(p), "--format=plaintext")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()
