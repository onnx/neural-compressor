# Neural Compressor

[![Python 3.11 and 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://github.com/onnx/neural-compressor)
[![Release](https://img.shields.io/badge/release-1.0-green)](https://github.com/onnx/neural-compressor/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/onnx/neural-compressor/blob/main/LICENSE)

Neural Compressor is an open-source Python library for compressing ONNX models. It provides post-training static and dynamic quantization, weight-only quantization, and SmoothQuant workflows that produce models for ONNX Runtime.

## Features

- Post-training static and dynamic quantization
- INT4/INT8 weight-only quantization with RTN, AWQ, and GPTQ
- Layer-wise quantization for reducing peak memory use
- Accuracy-aware tuning and examples for popular ONNX models

## Requirements

- Python 3.11 or 3.12
- An ONNX model and ONNX Runtime execution provider suitable for your model

See the [compatibility policy](./docs/Compatibility.md) for tested platforms and
minimum/current dependency combinations, and the [release notes](./docs/ReleaseNotes.md)
for changes planned for the next release. These requirements describe the current
source tree; the published v1.0 package predates these changes.

## Installation

Install the stable package from PyPI:

```shell
python -m pip install onnx-neural-compressor
```

To install from source with the test dependencies:

```shell
git clone https://github.com/onnx/neural-compressor.git
cd neural-compressor
python -m pip install -e ".[test]"
```

Run the test suite with:

```shell
python -m pytest -q test
```

See the [installation guide](./docs/installation_guide.md) for supported environments and hardware.

## Quick start

Dynamic quantization takes an existing ONNX model and writes a quantized model:

```python
from onnx_neural_compressor.quantization import config, quantize

quantize(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    quant_config=config.DynamicQuantConfig(),
)
```

For static quantization, provide a calibration data reader whose batches match the model's inputs. See the [quantization guide](./docs/quantization.md) for the reader interface and configuration options.

## Documentation and examples

| Guide | Description |
| --- | --- |
| [Quantization](./docs/quantization.md) | Static and dynamic post-training quantization |
| [Weight-only quantization](./docs/quantization_weight_only.md) | INT4/INT8 RTN, AWQ, and GPTQ |
| [Layer-wise quantization](./docs/quantization_layer_wise.md) | Quantize models layer by layer |
| [SmoothQuant](./docs/smooth_quant.md) | SmoothQuant workflow |
| [Autotuning](./docs/autotune.md) | Search quantization configurations |
| [Examples](./examples/) | End-to-end model examples |
| [Design](./docs/design.md) | Architecture and workflow |

## Contributing and support

- [Contribution guidelines](./docs/CONTRIBUTING.md)
- [Release process](./docs/Releasing.md)
- [Security policy](./SECURITY.md)
- [GitHub Issues](https://github.com/onnx/neural-compressor/issues) for bugs, feature requests, and questions
- [Email the maintainers](mailto:inc.maintainers@intel.com) with research and collaboration ideas
