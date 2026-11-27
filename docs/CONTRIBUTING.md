Contribution Guidelines
=======================
1. [Create Pull Request](#create-pull-request)
2. [Pull Request Checklist](#pull-request-checklist)
3. [Pull Request Template](#pull-request-template)
4. [Pull Request Acceptance Criteria](#pull-request-acceptance-criteria)
5. [Pull Request Status Checks Overview](#pull-request-status-checks-overview)
6. [Support](#support)
7. [Contributor Covenant Code of Conduct](#contributor-covenant-code-of-conduct)

## Create Pull Request
If you have improvements to Neural Compressor, send your pull requests for
[review](https://github.com/onnx/neural-compressor/pulls).
If you are new to GitHub, view the pull request [How To](https://help.github.com/articles/using-pull-requests/).
### Step-by-Step guidelines
- Star this repository using the button `Star` in the top right corner.
- Fork this Repository using the button `Fork` in the top right corner.
- Clone your forked repository to your pc.
`git clone "url to your repo"`
- Create a new branch for your modifications.
`git checkout -b new-branch`
- Add your files with `git add -A`, commit `git commit -s -m "This is my commit message"` and push `git push origin new-branch`.
- Create a [pull request](https://github.com/onnx/neural-compressor/pulls).

## Pull Request Checklist

Before sending your pull requests, follow the information below:

- Add unit tests in [Unit Tests](https://github.com/onnx/neural-compressor/tree/main/test) to cover the code you would like to contribute.
- Neural Compressor has adopted the [Developer Certificate of Origin](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin), you must agree to the terms of Developer Certificate of Origin by signing off each of your commits with `-s`, e.g. `git commit -s -m 'This is my commit message'`.

## Running Tests Locally

Install the package with its test dependencies, then run the test suite:

```shell
python -m pip install -e ".[test]"
python -m pytest -q test
```

Python 3.14 runs the portable test suite. Tests that exercise Hugging Face's
Optimum ONNX exporter run on Python 3.11 through 3.13 until its released
packages support Python 3.14. The release workflow separately verifies installed
dynamic and static quantization round trips on Python 3.14.

## Running Lint Locally

Install Ruff and run the same checks as CI:

```shell
python -m pip install -r requirements-lint.txt
ruff check .
ruff format --check .
```

Ruff can apply the available lint fixes and format the code:

```shell
ruff check --fix .
ruff format .
```

## Pull Request Template

Release managers should follow the [release process](Releasing.md) for version
changes, release branches, package validation, and publication.

See [PR template](/.github/pull_request_template.md)

## Pull Request Acceptance Criteria
- At least two approvals from reviewers

- All detected status checks pass

- All conversations solved

- Third-party dependency license compatible

## Pull Request Status Checks Overview
CI runs on GitHub-hosted Ubuntu runners for pull requests, pushes to `main`, and merge queues.

| Check | Scope | Pass criteria |
| --- | --- | --- |
| [DCO](https://github.com/apps/dco/) | Commit sign-offs | Use `git commit -s` |
| Lint | Python lint and formatting with Ruff | Required lint checks pass |
| Python 3.11 through Python 3.14 | [Test suite](../test), with portable tests on Python 3.14 | All tests pass; JUnit and coverage reports are uploaded |
| Minimum dependencies / Python 3.11 through Python 3.14 | Tests with declared runtime minimums | All tests and dependency checks pass |
| Coverage comparison | Python 3.12 line and branch coverage | Neither drops by more than 0.05 percentage points against the exact base commit |
| Model smoke tests | BERT dynamic/static QDQ; ResNet static QDQ/QOperator | Installed wheel quantizes, saves, reloads, and runs models on CPU |
| zizmor | GitHub workflow analysis | Workflow analysis passes |

Coverage reports are stored for 30 days. The comparison uses a successful `Tests`
push run for the PR or merge queue's exact base commit. If that commit has no
successful run or coverage artifact yet (including runs before this migration),
the job explicitly reports that no comparison was made. If a baseline artifact
exists but cannot be downloaded, the job fails; rerun the base push workflow or
refresh the PR base. API failures also fail the job.

Model smoke tests use small, seeded BERT and ResNet architectures with generated
weights and inputs, require no model downloads or private datasets, and execute
outside the checkout against a built wheel. They check graph conversion,
quantized operators, finite outputs, and numerical agreement on three held-out
inputs (`rtol=0.2`, `atol=0.02`). Reports are uploaded even when a model check fails.
The former private ImageNet/MRPC accuracy and Intel hardware throughput benchmarks
are retired; the smoke tests do not establish production accuracy or performance.

Both test workflows also support manual runs from the Actions tab. To run a model
smoke test locally after installing the package and `torch<2.5`:

```shell
python .github/ci/model_smoke.py --model bert --mode dynamic --output reports
```

When deploying this migration, repository administrators should disable the old
Azure service triggers and remove any legacy `onc ut test` / `onc model test`
required checks. Deleting their configuration files does not disconnect the
external service. Configure required GitHub checks after their first successful run.

## Support

Submit your questions, feature requests, and bug reports to the
[GitHub issues](https://github.com/onnx/neural-compressor/issues) page. You may also reach out to [Maintainers](mailto:inc.maintainers@intel.com).

## Contributor Covenant Code of Conduct

This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant Code of Conduct](./CODE_OF_CONDUCT.md).
