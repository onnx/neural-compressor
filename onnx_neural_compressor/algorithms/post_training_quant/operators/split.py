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
"""Split Operator."""

import onnx

from onnx_neural_compressor import constants, utility
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant.operators import base_op


@base_op.op_registry(op_types="Split", mode=[constants.STATIC_QUANT])
class SplitOperator(base_op.Operator):
    """Split Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(SplitOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        data_found, _, _, _, _ = self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
            return False
        if not all([self.quantizer.is_valid_quantize_weight(i) for i in node.input]):
            return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_outputs(self.node, direct_int8=True)
        node.name = node.name + "_quant"

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        parent = self.quantizer.model.get_parents(node)[0]
        children = self.quantizer.model.get_children(node)
        if (
            parent.op_type != "DequantizeLinear" or len(children) == 0 or not node.name.endswith("_quant")
        ):  # pragma: no cover
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parent = self.quantizer.model.get_parents(node)[0]
        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(quant_utils.attribute_to_kwarg(attribute))

        quantized_input_names = []
        quantized_input_names.append(parent.input[0])
        if len(node.input) > 1:  # pragma: no cover
            quantized_input_names.extend(node.input[1:])
        outputs = []
        input_name_to_nodes = self.quantizer.model.input_name_to_nodes()
        for output in node.output:
            if output in input_name_to_nodes:
                child = input_name_to_nodes[output][0]
                if child.op_type == "QuantizeLinear":
                    self.quantizer.remove_nodes.append(child)
                    outputs.append(child.output[0])
                else:  # pragma: no cover
                    outputs.append(output)
            else:  # pragma: no cover
                outputs.append(output + "_quantized")

        quantized_node = onnx.helper.make_node(node.op_type, quantized_input_names, outputs, node.name, **kwargs)
        self.quantizer.new_nodes.append(quantized_node)
        self.quantizer.remove_nodes.extend([parent, node])
