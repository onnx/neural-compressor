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
"""AveragePool Operator."""

import onnx

from onnx_neural_compressor import constants, utility
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant.operators import base_op


@base_op.op_registry(op_types="AveragePool", mode=[constants.STATIC_QUANT])
class PoolOperator(base_op.Operator):
    """AveragePool Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(PoolOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        if not self.quantizer.is_valid_quantize_weight(node.input[0]):
            return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        super().quantize()
        node.name = node.name + "_quant"

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)

        if len(children) == 0 or len(parents) == 0 or not node.name.endswith("_quant"):
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)

        if all([i.op_type == "DequantizeLinear" for i in parents]) and any(
            [i.op_type == "QuantizeLinear" for i in children]
        ):
            qlinear_output_name = node.output[0] + "_quantized"
            inputs = []
            inputs.extend(parents[0].input)
            inputs.extend([i for i in children if i.op_type == "QuantizeLinear"][0].input[1:])
            kwargs = {}
            for attribute in node.attribute:
                kwargs.update(quant_utils.attribute_to_kwarg(attribute))
            kwargs["domain"] = quant_utils.ms_domain
            qnode = onnx.helper.make_node("QLinear" + node.op_type, inputs, [qlinear_output_name], node.name, **kwargs)

            self.quantizer.remove_nodes.extend(parents)
            for child in children:
                if child.op_type == "QuantizeLinear":
                    self.quantizer.remove_nodes.append(child)
                    self.quantizer.model.replace_input_of_all_nodes(child.output[0], qnode.output[0])

            self.quantizer.new_nodes.append(qnode)
            self.quantizer.remove_nodes.append(node)
