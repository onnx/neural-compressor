"""Exercise the CI coverage gate's exit status and tolerance."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "ci" / "compare_coverage.py"


@pytest.mark.parametrize(
    "lines,branches,expected",
    [(9000, 8000, 0), (8995, 7995, 0), (8994, 8000, 1), (9000, 7994, 1), (10001, 8000, 1)],
)
def test_coverage_gate(tmp_path, lines, branches, expected):
    for name, line_count, branch_count in [("base", 9000, 8000), ("candidate", lines, branches)]:
        (tmp_path / f"{name}.xml").write_text(
            f'<coverage lines-covered="{line_count}" lines-valid="10000" '
            f'branches-covered="{branch_count}" branches-valid="10000"/>',
            encoding="utf-8",
        )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "candidate.xml"), str(tmp_path / "base.xml")],
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr


def test_missing_coverage_fails(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "missing.xml"), str(tmp_path / "base.xml")],
        capture_output=True,
    )
    assert result.returncode != 0
