# Release notes

## Unreleased

### Compatibility changes

- Supported Python versions are now **3.11 through 3.14** (`>=3.11,<3.15`). Python
  3.10 support is dropped. Python 3.15 and later are outside the validated range
  and are excluded by package metadata for this release.
- Runtime dependencies now have explicit minimum versions, tested together in CI.
  Current compatible dependencies are tested separately. See the
  [compatibility policy](Compatibility.md) for the version table and platform matrix.
- Python-specific minimums provide binary wheels on the supported interpreters.
  NumPy 1.26.4 remains supported on Python 3.11/3.12; Python 3.13/3.14 require
  NumPy >=2.3.5. Current compatible NumPy 2 versions are also tested.
  Transformers remains below version 5.
- `packaging` is now an explicit dependency because the library imports it directly.
- The test extra uses Optimum `>=1.27.0,<2` with PyTorch 2.4 on Python 3.11/3.12,
  and Optimum ONNX `>=0.1.0,<0.2` with PyTorch 2.9 on Python 3.13/3.14.
  Exporter constraints may select an older Transformers version than a runtime-only install.

### Quantization fixes

- Static and dynamic quantization now honor `reduce_range=True`, including
  per-channel weights. Previously the option was accepted but not forwarded to
  the quantizers. Explicit reduced-range requests now generate 7-bit weights,
  which can change quantized model contents and inference results. The default
  remains `reduce_range=False`. Revalidate accuracy when regenerating models that
  explicitly request reduced range. See [#56](https://github.com/onnx/neural-compressor/pull/56).

### Packaging and releases

- A single `VERSION_NUMBER` controls builds, installed package versions, and
  quantized-model producer versions. The old version-module import remains supported.
- Release CI validates wheels and source archives before manual publication through
  PyPI Trusted Publishing. See [the release process](Releasing.md).

The next release number has not been selected. The checked-in `1.0.0` is equivalent
to the already published `1.0` and cannot be published again as a new release.
