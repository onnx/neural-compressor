# Compatibility policy

This policy applies to the next release and the current `main` branch. Published
v1.0 has its own [release notes](https://github.com/onnx/neural-compressor/releases/tag/v1.0).

## Python and platforms

Supported interpreters are **CPython 3.11 and 3.12**. Package metadata enforces
`>=3.11,<3.13`. Python 3.10 is no longer supported. Python 3.13 and later need
validation before the supported range is extended.

Release CI installs both the wheel and source archive in clean environments on
Ubuntu 24.04 (x86-64), Windows Server 2022 (x86-64), and macOS 15 (ARM64), on both
supported Python versions. These checks exercise CPU dynamic and static
quantization, saved-model inference, dependency consistency, and package versions.
Other execution providers and hardware require separate model-specific validation.

## Runtime dependencies

The declared lower bounds are tested together as a minimum dependency set:

| Dependency | Minimum version |
| --- | --- |
| ONNX | 1.17.0 |
| ONNX Runtime | 1.20.1 |
| NumPy | 1.26.4 |
| SciPy | 1.12.0 |
| packaging | 24.2 |
| psutil | 5.9.8 |
| prettytable | 3.10.0 |
| py-cpuinfo | 9.0.0 |
| Pydantic | 2.7.4 |
| Transformers | 4.57.6, with an upper bound of `<5` |

The minimum set uses NumPy 1.26.4. Current dependency jobs also cover NumPy 2
where selected by the resolver. Both sets run the full unit suite on Linux and
the installed-artifact checks on all three platforms and both Python versions.

“Current” means the newest mutually compatible stable versions selected by pip
within package metadata and test-extra requirements at the time of the run.
Dependencies can impose additional compatibility constraints. This policy tests
these two sets; it does not certify every possible mixture of dependency versions.
CI runs `pip check` and records `pip freeze` for both sets so failures can be reproduced.

The minimum pins live in [`.github/constraints/minimum.txt`](../.github/constraints/minimum.txt).
They must match every direct runtime dependency's lower bound in `pyproject.toml`;
a release-policy test checks this. Test-only and transitive dependencies remain
resolver-selected. These constraints are a CI fixture, not a lockfile required
for normal package installation.

To reproduce the minimum full-suite job from a fresh Python 3.11 or 3.12 environment:

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
