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
"""Base Operator."""

from onnx_neural_compressor import constants, quantization

OPERATORS = {
    "dynamic_quant": {},
    "static_quant": {},
}


def op_registry(op_types, mode):
    """The class decorator used to register all Operator subclasses."""

    def decorator_op(cls):
        assert cls.__name__.endswith(
            "Operator"
        ), "The name of subclass of Operator should end with 'Operator' substring."
        for item in mode:
            if cls.__name__[: -len("Operator")] in OPERATORS[item]:  # pragma: no cover
                raise ValueError("Cannot have two operators with the same name for {} mode.".format(item))
                break
        for single_op_type in [op_type.strip() for op_type in op_types.split(",")]:
            for item in mode:
                OPERATORS[item][single_op_type] = cls
        return cls

    return decorator_op


class Operator(object):
    """Base Operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """Initialization."""
        self.quantizer = onnx_quantizer
        self.node = onnx_node
        node_name = self.node.name.split("_quant")[0]
        if node_name in self.quantizer.config:
            self.dtype = self.quantizer.config[node_name]
        self.disable_qdq_for_node_output = (
            True if onnx_node.op_type in onnx_quantizer.optypes_to_exclude_output_quant else False
        )
        self.per_channel = False
        self.calibrate_method = 0  # minmax
        self.weight_sym = True
        self.weight_dtype = None
        self.activation_dtype = None
        self.activation_sym = False
        if node_name in self.quantizer.config:
            if self.quantizer.config[node_name] not in self.quantizer.fallback_list:
                self.per_channel = self.quantizer.config[node_name]["per_channel"]
                self.calibrate_method = self.quantizer.config[node_name]["calibrate_method"]
                self.weight_sym = self.quantizer.config[node_name]["weight_sym"]
                self.weight_dtype = self.quantizer.config[node_name]["weight_type"]
                self.activation_dtype = self.quantizer.config[node_name]["activation_type"]
                self.activation_sym = self.quantizer.config[node_name]["activation_sym"]

    def quantize_check(self):
        """Check if quantizaion can be done."""
        return True

    def quantize(self):
        """Do quantizaion."""
        node = self.node
        self.quantizer.quantize_inputs(node)
        if not self.disable_qdq_for_node_output or self.quantizer.mode != constants.DYNAMIC_QUANT:
            self.quantizer.quantize_outputs(node)

    def convert_check(self):
        """Check if conversion can be done."""
        node = self.node

        if not node.name.endswith("_quant"):
            return False
        return True

    def convert(self):
        """Convert to QOperator format."""
        return
