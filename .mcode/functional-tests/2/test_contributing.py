import os, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VENV_PY = os.path.join(os.environ.get("MCODE_DIR", ""), "scratch", "venv39", "bin", "python")


def run(args, timeout=600):
    return subprocess.run([VENV_PY] + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def test_contributing_running_tests_section():
    text = open(os.path.join(ROOT, "CONTRIBUTING.rst")).read()
    assert "Running tests\n---" in text
    section = text.split("Running tests\n", 1)[1].split("\n", 1)[1].strip().splitlines()
    assert len(section) == 3


def test_documented_full_suite():
    r = run(["-m", "pytest", "-p", "no:typeguard", "-q"])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]


def test_documented_single_module():
    r = run(["-m", "pytest", "-p", "no:typeguard", "tests/test_parsers.py"])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]


def test_cli_help_still_works():
    r = run(["-m", "sumy", "--help"])
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()
