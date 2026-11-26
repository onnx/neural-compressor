"""Keep the minimum test environment aligned with the installable package contract."""

import tomllib
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("python_version", ["3.11", "3.12", "3.13", "3.14"])
def test_minimum_constraints_match_runtime_lower_bounds(python_version):
    environment = {
        **default_environment(),
        "python_version": python_version,
        "python_full_version": python_version + ".0",
    }
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    active = [
        req
        for req in map(Requirement, project["dependencies"])
        if req.marker is None or req.marker.evaluate(environment)
    ]
    dependencies = {canonicalize_name(req.name): req for req in active}
    assert len(active) == len(dependencies), "Overlapping runtime dependency markers"
    lines = (ROOT / ".github/constraints/minimum.txt").read_text(encoding="utf-8").splitlines()
    pins = [Requirement(line) for line in lines if line.strip() and not line.startswith("#")]
    pins = [pin for pin in pins if pin.marker is None or pin.marker.evaluate(environment)]
    assert len(pins) == len(dependencies)
    assert {canonicalize_name(pin.name) for pin in pins} == dependencies.keys()
    for pin in pins:
        requirement = dependencies[canonicalize_name(pin.name)]
        lower_bounds = [bound.version for bound in requirement.specifier if bound.operator == ">="]
        assert len(lower_bounds) == 1, f"Expected one inclusive minimum: {requirement}"
        assert str(pin.specifier) == f"=={lower_bounds[0]}", f"Minimum drift: {pin} vs {requirement}"
        assert lower_bounds[0] in requirement.specifier
