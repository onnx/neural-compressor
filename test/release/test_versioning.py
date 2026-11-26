"""Release publication guards and installed version compatibility."""

import datetime
import importlib.metadata
import runpy
import tomllib
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
prepare_version = runpy.run_path(str(ROOT / ".github/release_tools/prepare.py"))["prepare_version"]


def prepare(**changes):
    arguments = dict(
        version="1.1.0",
        mode="release",
        publish="pypi",
        event="workflow_dispatch",
        repository="onnx/neural-compressor",
        ref="refs/heads/rel-1.1.0",
        today=datetime.date(2026, 11, 26),
    )
    arguments.update(changes)
    return prepare_version(**arguments)


@pytest.mark.parametrize("version", ["1.1.0", "1.1.0rc1", "1.1.0rc12"])
@pytest.mark.parametrize("publish", ["testpypi", "pypi"])
def test_release_and_candidate_versions(version, publish):
    assert prepare(version=version, publish=publish) == version


def test_preview_is_explicit_and_date_based():
    assert prepare(mode="preview", publish="none", event="pull_request") == "1.1.0.dev20261126"


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "preview"},
        {"event": "pull_request"},
        {"event": "push"},
        {"event": "schedule"},
        {"repository": "someone/neural-compressor"},
        {"ref": "refs/heads/main"},
        {"ref": "refs/heads/rel-1.1.1"},
        {"ref": "refs/tags/v1.1.0"},
        {"version": "1.1.0+local"},
        {"version": "1.1.0.dev1"},
        {"version": "1.1.0rc0"},
        {"version": "1.1"},
        {"version": "1.1.0\nextra=value"},
    ],
)
def test_invalid_publish_requests_are_rejected(changes):
    with pytest.raises(ValueError):
        prepare(**changes)


def test_dry_release_build_is_allowed_on_main():
    assert prepare(publish="none", event="push", ref="refs/heads/main") == "1.1.0"


def test_runtime_uses_distribution_metadata():
    with mock.patch("importlib.metadata.version", return_value="1.1.0rc2"):
        module = runpy.run_path(str(ROOT / "onnx_neural_compressor/__init__.py"))
    assert module["__version__"] == "1.1.0rc2"


def test_uninstalled_source_has_unknown_version():
    with mock.patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
        module = runpy.run_path(str(ROOT / "onnx_neural_compressor/__init__.py"))
    assert module["__version__"] == "unknown"


def test_supported_python_versions_and_declarative_version_source():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    project = pyproject["project"]
    assert project["requires-python"] == ">=3.11"
    assert {
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    } <= set(project["classifiers"])
    assert "optimum[onnxruntime]>=2; python_version < '3.14'" in project["optional-dependencies"]["test"]
    assert "torch; python_version < '3.14'" in project["optional-dependencies"]["test"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {"file": ["VERSION_NUMBER"]}
    assert not (ROOT / "setup.py").exists()
