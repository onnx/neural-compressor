# Installation

1. [Installation](#installation)

    1.1. [Prerequisites](#prerequisites)

    1.2. [Install from Binary](#install-from-binary)

    1.3. [Install from Source](#install-from-source)

2. [System Requirements](#system-requirements)

   2.1. [Validated Hardware Environment](#validated-hardware-environment)

   2.2. [Validated Software Environment](#validated-software-environment)

## Installation
### Prerequisites
You can install Neural Compressor from binary or source.

The following prerequisites and requirements must be satisfied for a successful installation:

- Python version: 3.11 or 3.12

Package metadata restricts this source tree and the next release to Python
`>=3.11,<3.13`. See the [compatibility policy](Compatibility.md) for dependency
minimums and the [release notes](ReleaseNotes.md) for migration details. The
published v1.0 package has different requirements.

### Install from Binary
  ```Shell
  # install stable basic version from pypi
  pip install onnx-neural-compressor
  ```

### Install from Source

  ```Shell
  git clone https://github.com/onnx/neural-compressor.git
  cd neural-compressor
  pip install .
  ```

## System Requirements

### Validated Hardware Environment
#### Neural Compressor supports CPUs based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):

* Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, Cooper Lake, Ice Lake, and Sapphire Rapids)
* Intel Xeon CPU Max Series (formerly Sapphire Rapids HBM)
* Intel Core Ultra Processors (Meteor Lake, Lunar Lake)

### Validated Software Environment

* OS versions: Ubuntu 24.04, Windows Server 2022, macOS 15 (CPU artifact tests)
* Python version: 3.11 or 3.12
* ONNX Runtime version: 1.20.1 minimum; current compatible versions also tested
* Minimum and current dependency sets: see the [compatibility matrix](Compatibility.md)
