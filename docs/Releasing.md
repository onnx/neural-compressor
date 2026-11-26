# Releasing ONNX Neural Compressor

This process follows [ONNX's release conventions](https://github.com/onnx/onnx/blob/main/docs/OnnxReleases.md):
one `VERSION_NUMBER` file, `rel-X.Y.Z` branches, `X.Y.ZrcN` candidates, manual
publication, protected environments, and PyPI Trusted Publishing. The package is
pure Python, so a single universal wheel and source archive serve all platforms.

## Version source

`VERSION_NUMBER` is the only maintained package version. `setup.py` reads it
without Git or importing the package, and `MANIFEST.in` includes it in the source
archive. Builds from a checkout and from an extracted source archive therefore
use the same version. Installed callers can use `onnx_neural_compressor.__version__`;
the existing `onnx_neural_compressor.version.__version__` import remains supported.
Both read installed distribution metadata. Uninstalled source reports `unknown`.

The initial `1.0.0` is equivalent to the previously published `1.0`; this setup
change does not select a new release or make that version publishable again.

## One-time administrator setup

Create GitHub environments `testpypi-release` and `pypi-release`, with required
release reviewers and deployment branch restrictions allowing only `rel-*`.
Configure branch protection for release branches; it is not created by the YAML.

Register a Trusted Publisher on the `onnx-neural-compressor` project on both
TestPyPI and PyPI with:

| Field | Value |
| --- | --- |
| Owner | `onnx` |
| Repository | `neural-compressor` |
| Workflow filename | `create_release.yml` |
| Environment | `testpypi-release` or `pypi-release`, respectively |

See [PyPI's Trusted Publishing instructions](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
No PyPI token or password is stored in GitHub. These settings need administrator
access and must be configured before requesting publication.

## Prepare a release

1. Agree on `X.Y.Z` and prepare notes covering changes, compatibility changes,
   Python/dependency support, and known issues. Link the relevant PRs.
2. Cut `rel-X.Y.Z` from the tested `main` commit. For example, `rel-1.1.0`.
3. Set `VERSION_NUMBER` on that branch to `X.Y.Zrc1` through a reviewed PR.
4. Advance `main` to the next planned version in a separate PR.
5. Apply fixes to `main` and cherry-pick release fixes onto the release branch.

## Build and validate

The **Create Releases** workflow runs for PRs, pushes to `main` / `rel-*`, merge
queues, weekly on Monday at 00:00 UTC, and manual dispatches. Automatic runs only
build preview artifacts, appending `.devYYYYMMDD` in UTC to `VERSION_NUMBER` in
the temporary checkout. They never publish and do not commit the version change.
Preview artifacts keep the `onnx-neural-compressor` distribution name; no separate
weekly PyPI project is needed.

For a release candidate or final build, select the release branch and choose:

- `build_mode: release` to build the exact checked-in version.
- `publish: none` for a dry run (the default publication choice).

The workflow builds the source archive and then builds its wheel from that archive,
checks metadata with Twine and verifies matching versions. It installs each artifact
into a separate clean environment on Linux, Windows, and macOS with Python 3.11 and
3.12, runs `pip check`, and verifies version imports and dynamic/static quantization
round trips outside the checkout. Download both files from the `distributions`
artifact. Artifacts are retained for 30 days.

For a local exact-version build:

```shell
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

## Publish candidates and the final release

1. Run **Create Releases** from `rel-X.Y.Z`, with `build_mode: release` and
   `publish: testpypi`. Review the successful checks and approve the environment
   deployment. The branch suffix must match the version, with an optional `rcN`.
2. Validate the candidate with downstream users. Candidates can also be published
   to PyPI by selecting `publish: pypi`, as in ONNX's release process.
3. Fix problems and increment the candidate number before publishing another build.
4. When ready, remove `rcN` in a reviewed PR on the release branch. Run a dry build,
   draft release notes, and create the immutable `vX.Y.Z` tag at the tested commit.
5. Dispatch with `build_mode: release` and `publish: pypi`. After all artifact checks
   pass, approve the `pypi-release` deployment. The job attests and uploads the exact
   artifacts tested in that workflow run; it does not rebuild or check out code.
6. Publish the GitHub release notes and announce the release through the project's
   usual channels. Merge any release-only fixes back into `main`.

Publication is accepted only for manual release builds in `onnx/neural-compressor`
on a matching release branch. Preview publication and mismatched versions fail
before the build. PyPI versions cannot be overwritten: use a new version for a
replacement. Never move a published release tag. Workflow dispatch itself does not
create tags, GitHub releases, release branches, environments, or publisher registrations.
