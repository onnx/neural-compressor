"""Keep the minimum test environment aligned with the installable package contract."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]


def test_minimum_constraints_match_runtime_lower_bounds():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = {canonicalize_name(req.name): req for req in map(Requirement, project["dependencies"])}
    lines = (ROOT / ".github/constraints/minimum.txt").read_text(encoding="utf-8").splitlines()
    pins = [Requirement(line) for line in lines if line.strip() and not line.startswith("#")]
    assert len(pins) == len(dependencies)
    assert {canonicalize_name(pin.name) for pin in pins} == dependencies.keys()
    for pin in pins:
        requirement = dependencies[canonicalize_name(pin.name)]
        lower_bounds = [bound.version for bound in requirement.specifier if bound.operator == ">="]
        assert len(lower_bounds) == 1, f"Expected one inclusive minimum: {requirement}"
        assert str(pin.specifier) == f"=={lower_bounds[0]}", f"Minimum drift: {pin} vs {requirement}"
        assert lower_bounds[0] in requirement.specifier
