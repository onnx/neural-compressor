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
"""Gemm Operator."""

import onnx

from onnx_neural_compressor import constants, logger
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant.operators import base_op


@base_op.op_registry(op_types="Gemm", mode=[constants.STATIC_QUANT])
class GemmOperator(base_op.Operator):
    """Gemm Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(GemmOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        if len(node.input) == 3 and not quant_utils.find_by_name(node.input[2], self.quantizer.model.initializer()):

            logger.warning(
                "Bias of Gemm node '{}' is not constant. "
                "Exclude this node can get better performance.".format(node.name)
            )
            if self.quantizer.quant_format != "qdq":
                return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node, [0])
        if self.per_channel and quant_utils.find_by_name(node.input[1], self.quantizer.model.initializer()):
            self.quantizer.quantize_weights_per_channel(
                node, [1], self.weight_dtype, self.weight_sym, 0 if quant_utils.is_B_transposed(node) else 1
            )
        else:
            self.quantizer.quantize_inputs(node, [1])

        if len(node.input) == 3 and quant_utils.find_by_name(node.input[2], self.quantizer.model.initializer()):
            self.quantizer.quantize_bias_tensor(node)
            beta_attribute = [attr for attr in node.attribute if attr.name == "beta"]
            if len(beta_attribute):
                beta_attribute[0].f = 1.0

        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_outputs(node)
        node.name = node.name + "_quant"

    def convert(self):
        """Convert to QOperator format."""
        node = self.node

        parents = self.quantizer.model.get_parents(node)
        qgemm_inputs = []
        for parent in parents[:-1]:
            qgemm_inputs.extend(parent.input)
        qgemm_inputs.append(parents[-1].input[0])

        kwargs = {}
        for attribute in node.attribute:
            if attribute.name != "beta":
                kwargs.update(quant_utils.attribute_to_kwarg(attribute))
                kwargs["domain"] = quant_utils.ms_domain

        qgemm_output = node.output[0]
        if not self.disable_qdq_for_node_output:
            child = self.quantizer.model.get_children(node)[0]
            self.quantizer.remove_nodes.append(child)
            qgemm_output = child.output[0]
            qgemm_inputs.extend(child.input[1:])
        qgemm_node = onnx.helper.make_node("QGemm", qgemm_inputs, [qgemm_output], node.name, **kwargs)

        self.quantizer.new_nodes.append(qgemm_node)
        self.quantizer.remove_nodes.extend(parents)
        self.quantizer.remove_nodes.append(node)
