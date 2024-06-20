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
"""GlobalAveragePool Operator."""

import onnx

from onnx_neural_compressor.algorithms.post_training_quant.operators import base_op
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor import constants
from onnx_neural_compressor import utility


@base_op.op_registry(op_types="GlobalAveragePool", mode=[constants.STATIC_QUANT])
class GlobalAveragePoolOperator(base_op.Operator):
    """GlobalAveragePool Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(GlobalAveragePoolOperator, self).__init__(onnx_quantizer, onnx_node)

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        children = self.quantizer.model.get_children(node)
        if len(children) == 0:  # pragma: no cover
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parent = self.quantizer.model.get_parents(node)[0]
        child = self.quantizer.model.get_children(node)[0]

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(quant_utils.attribute_to_kwarg(attribute))
        kwargs["domain"] = quant_utils.ms_domain
        kwargs["channels_last"] = 0

        inputs = parent.input
        inputs.extend(child.input[1:])

        qnode = onnx.helper.make_node("QLinear" + node.op_type, inputs, child.output, node.name + "_quant", **kwargs)
        self.quantizer.new_nodes += [qnode]
        self.quantizer.remove_nodes.append(child)
        self.quantizer.remove_nodes.append(parent)
        self.quantizer.remove_nodes.append(node)
