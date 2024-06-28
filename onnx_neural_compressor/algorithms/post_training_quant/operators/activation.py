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
"""Activation operator."""

import onnx

from onnx_neural_compressor import constants, utility
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant.operators import base_op


@base_op.op_registry(op_types="LeakyRelu, Sigmoid", mode=[constants.STATIC_QUANT])
class ActivationOperator(base_op.Operator):
    """Activation operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(ActivationOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        data_found, _, _, _, _ = self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
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

        children = self.quantizer.model.get_children(node)
        if len(children) == 0 or not node.name.endswith("_quant"):
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parent = self.quantizer.model.get_parents(node)[0]
        child = self.quantizer.model.get_children(node)[0]

        inputs = []
        inputs.extend(parent.input)
        inputs.extend(child.input[1:])

        qlinear_activation_output = child.output[0]
        kwargs = {}
        for attribute in node.attribute:  # pragma: no cover
            kwargs.update(quant_utils.attribute_to_kwarg(attribute))
        kwargs["domain"] = quant_utils.ms_domain

        qlinear_activation_node = onnx.helper.make_node(
            "QLinear" + node.op_type, inputs, [qlinear_activation_output], node.name, **kwargs
        )

        self.quantizer.new_nodes.append(qlinear_activation_node)
        self.quantizer.remove_nodes.extend([parent, child, node])


@base_op.op_registry(op_types="Relu, Clip", mode=[constants.STATIC_QUANT])
class RemovableActivationOperator(base_op.Operator):
    """Removable activation operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(RemovableActivationOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        if node.input[0] not in self.quantizer.quantized_value_map:
            return False
        return True

    def quantize(self):
        """Do quantization."""
        node = self.node
        if node.output[0] in [i.name for i in self.quantizer.model.model.graph.output]:
            self.quantizer.dequantize_tensor(node, node.input[0])
        else:
            self.quantizer.model.replace_input_of_all_nodes(node.output[0], node.input[0])
            self.quantizer.remove_nodes.append(node)


@base_op.op_registry(
    op_types="Softmax, BiasGelu, Elu, Exp, FastGelu, Gelu, Softplus, Tanh", mode=[constants.STATIC_QUANT]
)
class Float16ActivationOperator(base_op.Operator):
    """Float16 Activation operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(Float16ActivationOperator, self).__init__(onnx_quantizer, onnx_node)
