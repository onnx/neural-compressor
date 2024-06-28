# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ArgMax operator."""

from onnx_neural_compressor import constants, utility
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant.operators import base_op


@base_op.op_registry(op_types="ArgMax", mode=[constants.STATIC_QUANT])
class ArgMaxOperator(base_op.Operator):
    """ArgMax operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(ArgMaxOperator, self).__init__(onnx_quantizer, onnx_node)

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        return True

    def convert(self):
        """Convert to quantized format."""
        node = self.node
        origin_name = node.input[0].split("_argmax_node")[0]

        if origin_name in self.quantizer.quantized_value_map:
            node.name = node.name + "_quant"
