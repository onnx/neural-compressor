# Release notes

## Unreleased

### Compatibility changes

- Supported Python versions are now **3.11 through 3.14**. Python 3.10 support is
  dropped. Package metadata requires `>=3.11`; later Python versions need
  validation before they are advertised as supported.
- Runtime dependencies now have explicit minimum versions, tested together in CI.
  Current compatible dependencies are tested separately. See the
  [compatibility policy](Compatibility.md) for the version table and platform matrix.
- Python-specific minimums provide binary wheels on the supported interpreters.
  NumPy 1.26.4 remains supported on Python 3.11/3.12; Python 3.13/3.14 require
  NumPy >=2.3.5. Current compatible NumPy 2 versions are also tested.
  Transformers remains below version 5.
- `packaging` is now an explicit dependency because the library imports it directly.
- Python 3.11-3.13 run the full suite with Optimum `<2` and PyTorch `<2.9`.
  Python 3.14 runs the portable suite and installed quantization checks while
  released Optimum exporters lack Python 3.14 support. Exporter constraints may
  select an older Transformers version than a runtime-only install.

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
