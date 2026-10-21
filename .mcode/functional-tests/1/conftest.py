# -*- coding: utf8 -*-
"""Shared fixtures for sumy functional tests.

sumy is a library + thin CLI tool (console scripts `sumy` / `sumy_eval`),
not a network service, so "functional" here means: invoke the real,
installed console scripts (and the real importable library API) and check
their actual output/exit codes.
"""

import shutil
import sys
from pathlib import Path

import pytest

# .mcode/functional-tests/1/conftest.py -> repo root is 3 parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
VENV_BIN = REPO_ROOT / ".venv" / "bin"


def _resolve_bin(name):
    """Prefer the project's own dedicated venv; fall back to PATH."""
    candidate = VENV_BIN / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    pytest.skip(f"Could not locate the '{name}' console script (expected at {candidate})")


SUMY_BIN = _resolve_bin("sumy") if (VENV_BIN / "sumy").exists() or shutil.which("sumy") else None
SUMY_EVAL_BIN = _resolve_bin("sumy_eval") if (VENV_BIN / "sumy_eval").exists() or shutil.which("sumy_eval") else None

SAMPLE_EN_TEXT = (
    "Artificial intelligence has transformed the way people interact with "
    "technology in daily life. Many industries, including healthcare, "
    "finance, and transportation, now rely heavily on machine learning "
    "models to automate decisions. These models are trained on large "
    "datasets that capture patterns humans might otherwise overlook. "
    "However, the rapid adoption of AI also raises important questions "
    "about fairness, privacy, and accountability. Researchers are actively "
    "studying how to make these systems more transparent and explainable. "
    "At the same time, companies are racing to deploy new AI products "
    "before fully understanding their long term societal impact. "
    "Policymakers around the world are now debating how best to regulate "
    "this powerful and fast moving technology. Education systems are also "
    "adapting, teaching students to work alongside intelligent tools "
    "rather than compete with them. Despite the challenges, many experts "
    "remain optimistic that careful stewardship of AI can lead to "
    "significant benefits for humanity. The next decade will likely "
    "determine whether this optimism is justified.\n"
)

SAMPLE_DE_TEXT = (
    "Die Digitalisierung verändert die deutsche Wirtschaft seit vielen "
    "Jahren grundlegend. Viele Unternehmen investieren stark in neue "
    "Technologien, um wettbewerbsfähig zu bleiben. Besonders die "
    "Automobilindustrie steht vor großen Herausforderungen durch die "
    "Umstellung auf Elektromobilität. Gleichzeitig entstehen neue "
    "Arbeitsplätze in der Softwareentwicklung und im Bereich künstliche "
    "Intelligenz. Experten warnen jedoch, dass der Wandel auch Risiken für "
    "traditionelle Berufe mit sich bringt. Die Bundesregierung hat mehrere "
    "Förderprogramme aufgelegt, um den digitalen Wandel zu unterstützen. "
    "Bildungseinrichtungen passen ihre Lehrpläne an, um Studierende besser "
    "auf die neue Arbeitswelt vorzubereiten. Trotz aller Unsicherheiten "
    "bleibt die Mehrheit der Unternehmen optimistisch für die Zukunft.\n"
)

SAMPLE_HTML = (
    "<!DOCTYPE html>\n<html>\n<head><title>Sample Article About Space "
    "Exploration</title></head>\n<body>\n<article>\n<h1>Sample Article "
    "About Space Exploration</h1>\n<p>Space exploration has entered a new "
    "era of international collaboration and private investment. Companies "
    "around the world are now building reusable rockets to cut the cost of "
    "reaching orbit. Scientists hope that lower launch costs will enable "
    "more ambitious missions to the Moon and Mars. Meanwhile, "
    "international space agencies continue to operate the International "
    "Space Station as a platform for long duration research. Private "
    "astronauts have also begun flying to orbit, marking a shift away from "
    "purely government sponsored spaceflight. Many experts believe the "
    "next major milestone will be a sustained human presence on the Moon. "
    "Looking further ahead, a crewed mission to Mars remains the ultimate "
    "goal for several national space programs. These efforts continue to "
    "inspire new generations of engineers and scientists across the "
    "globe.</p>\n</article>\n</body>\n</html>\n"
)

CZECH_FIXTURE = REPO_ROOT / "tests" / "data" / "articles" / "prevko_cz_1.txt"


@pytest.fixture(scope="session")
def sumy_bin():
    if SUMY_BIN is None:
        pytest.skip("sumy console script not found")
    return SUMY_BIN


@pytest.fixture(scope="session")
def sumy_eval_bin():
    if SUMY_EVAL_BIN is None:
        pytest.skip("sumy_eval console script not found")
    return SUMY_EVAL_BIN


@pytest.fixture(scope="session")
def sample_en_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("ft_sumy_data") / "sample_en.txt"
    path.write_text(SAMPLE_EN_TEXT, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def sample_de_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("ft_sumy_data") / "sample_de.txt"
    path.write_text(SAMPLE_DE_TEXT, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def sample_html_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("ft_sumy_data") / "sample.html"
    path.write_text(SAMPLE_HTML, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def reference_summary_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("ft_sumy_data") / "reference_summary.txt"
    path.write_text(
        "Artificial intelligence has transformed the way people interact "
        "with technology in daily life.\n"
        "Many industries, including healthcare, finance, and "
        "transportation, now rely heavily on machine learning models to "
        "automate decisions.\n",
        encoding="utf-8",
    )
    return path
