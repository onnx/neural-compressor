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
"""Direct8Bit Operator."""

from onnx_neural_compressor import constants, utility
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant.operators import base_op


@base_op.op_registry(
    op_types="Reshape, Transpose, Squeeze, Unsqueeze, Flatten, Expand, Slice, "
    "SpaceToDepth, DepthToSpace, Upsample, Tile, CenterCropPad",
    mode=[constants.STATIC_QUANT],
)
class Direct8BitOperator(base_op.Operator):
    """Direct8Bit Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        super(Direct8BitOperator, self).__init__(onnx_quantizer, onnx_node)

    def quantize_check(self):
        """Check if quantizaion can be done."""
        node = self.node
        if not self.quantizer.is_valid_quantize_weight(node.input[0]):
            return False
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(self.node, [0], initializer_use_weight_qType=False, direct_int8=True)
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_outputs(self.node, direct_int8=True)
        node.name = node.name + "_quant"

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node
        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)
        if (len(children) == 0 and len(parents) == 0) or not node.name.endswith("_quant"):
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        node = self.node
        parents = self.quantizer.model.get_parents(node)
        children = self.quantizer.model.get_children(node)
        if any([i.op_type == "DequantizeLinear" for i in parents]) and any(
            [i.op_type == "QuantizeLinear" for i in children]
        ):
            for parent in parents:
                if parent.op_type == "DequantizeLinear":
                    # make sure parent DequantizeLinear of input 0 is not used by other ops
                    if len(self.quantizer.model.get_children(parent)) == 1 and not self.quantizer.model.is_graph_output(
                        parents[0].output[0]
                    ):
                        self.quantizer.remove_nodes.append(parent)
                    self.node.input[0] = parent.input[0]
                    break
            for child in children:
                if child.op_type == "QuantizeLinear":
                    self.quantizer.remove_nodes.append(child)
                    self.quantizer.model.replace_input_of_all_nodes(child.output[0], node.output[0] + "_quantized")
            node.output[0] = node.output[0] + "_quantized"
