# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum

import onnx


class QuantType(enum.Enum):  # pragma: no cover
    """Represent QuantType value."""

    QInt8 = 0
    QUInt8 = 1
    QInt4 = 4
    QUInt4 = 5

    @property
    def tensor_type(self):
        if self == QuantType.QInt8:
            return onnx.TensorProto.INT8
        if self == QuantType.QUInt8:
            return onnx.TensorProto.UINT8
        if self == QuantType.QInt8:
            return onnx.TensorProto.INT4
        if self == QuantType.QUInt4:
            return onnx.TensorProto.UINT4
        raise ValueError(f"Unexpected value qtype={self!r}.")


class QuantFormat(enum.Enum):
    QOperator = 0
    QDQ = 1


class CalibrationMethod(enum.Enum):
    MinMax = 0
    Entropy = 1
    Percentile = 2
    Distribution = 3
