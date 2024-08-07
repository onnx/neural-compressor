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

- Python version: 3.8 or 3.9 or 3.10 or 3.11

### Install from Binary
  ```Shell
  # install stable basic version from pypi
  pip install onnx-neural-compressor
  ```

### Install from Source

  ```Shell
  git clone https://github.com/onnx/neural-compressor.git
  cd neural-compressor
  pip install -r requirements.txt
  pip install .
  ```

## System Requirements

### Validated Hardware Environment
#### Neural Compressor supports CPUs based on [Intel 64 architecture or compatible processors](https://en.wikipedia.org/wiki/X86-64):

* Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, Cooper Lake, Ice Lake, and Sapphire Rapids)
* Intel Xeon CPU Max Series (formerly Sapphire Rapids HBM)
* Intel Core Ultra Processors (Meteor Lake, Lunar Lake)

### Validated Software Environment

* OS version: CentOS 8.4, Ubuntu 22.04
* Python version: 3.10
* ONNX Runtime version: 1.18.1
