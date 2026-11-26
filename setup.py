# Copyright (c) ONNX Neural Compressor Project Contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from setuptools import setup

setup(version=Path(__file__).with_name("VERSION_NUMBER").read_text(encoding="utf-8").strip())
