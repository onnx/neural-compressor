# Compatibility policy

This policy applies to the next release and the current `main` branch. Published
v1.0 has its own [release notes](https://github.com/onnx/neural-compressor/releases/tag/v1.0).

## Python and platforms

Supported interpreters are **CPython 3.11, 3.12, 3.13, and 3.14** (standard builds).
Package metadata requires `>=3.11`. Python 3.10 is no longer supported. Later
Python versions are not blocked by an upper bound, but need validation before
they are included in the supported range.

Release CI installs both the wheel and source archive in clean environments on
Ubuntu 24.04 (x86-64), Windows Server 2022 (x86-64), and macOS 15 (ARM64), on all four
supported Python versions in each dependency set. These checks exercise CPU dynamic and static
quantization, saved-model inference, dependency consistency, and package versions.
Other execution providers and hardware require separate model-specific validation.

## Runtime dependencies

The declared lower bounds are tested together as a minimum dependency set:

| Dependency | Python 3.11-3.12 minimum | Python 3.13-3.14 minimum |
| --- | --- | --- |
| ONNX | 1.17.0 | 1.20.1 |
| ONNX Runtime | 1.20.1 | 1.24.2 |
| NumPy | 1.26.4 | 2.3.5 |
| SciPy | 1.12.0 | 1.16.3 |
| packaging | 24.2 | 24.2 |
| psutil | 5.9.8 | 7.1.0 |
| prettytable | 3.10.0 | 3.10.0 |
| py-cpuinfo | 9.0.0 | 9.0.0 |
| Pydantic | 2.7.4 | 2.12.5 |
| Transformers | 4.53.3, `<5` | 4.53.3, `<5` |

Python-specific dependency markers select a baseline with binary wheels for each
interpreter. Python 3.11/3.12 minimum jobs retain NumPy 1.26.4; Python 3.13/3.14 use
NumPy 2.3.5 or newer. Current dependency jobs also cover NumPy 2
where selected by the resolver. Both sets run unit tests on Linux and the
installed-artifact checks on all three platforms and all four Python versions.

“Current” means the newest mutually compatible stable versions selected by pip
within package metadata and test-extra requirements at the time of the run.
Dependencies can impose additional compatibility constraints. This policy tests
these two sets; it does not certify every possible mixture of dependency versions.
CI runs `pip check` and records `pip freeze` for both sets so failures can be reproduced.

Python 3.11-3.13 run the full suite with Optimum `<2`, PyTorch `<2.9`, and SymPy
in the test extra. Python 3.14 runs the portable suite: released Optimum exporters
do not yet support that interpreter, so the exporter-dependent tests listed in
`test/conftest.py` are not collected there. Its installed-artifact jobs still test
dynamic/static quantization and inference on all three platforms with both
dependency sets. This is a test-tool limitation, not an exclusion of Python 3.14
from package support.

Optimum further constrains the Transformers version selected for the full-suite
jobs. Installed-artifact jobs use only runtime dependencies and also test
the newest compatible Transformers 4.x independently of the export tooling.

The minimum pins live in [`.github/constraints/minimum.txt`](../.github/constraints/minimum.txt).
They must match every direct runtime dependency's lower bound in `pyproject.toml`;
a release-policy test checks each supported Python version. Test-only and transitive dependencies remain
resolver-selected. These constraints are a CI fixture, not a lockfile required
for normal package installation.

To reproduce the minimum full-suite job from a fresh supported Python environment:

```shell
python -m pip install --constraint .github/constraints/minimum.txt -e ".[test]"
python -m pip check
python -m pytest -q test
```

Omit `--constraint .github/constraints/minimum.txt` to reproduce the current job.

## Updating support

Changes to supported Python versions or dependency bounds must update package
metadata, minimum constraints, this document, and release notes together. Extend
the unit and installed-artifact matrices before advertising a new Python version.
Every release candidate must pass both dependency sets. Record any support changes
in [the release notes](ReleaseNotes.md).
