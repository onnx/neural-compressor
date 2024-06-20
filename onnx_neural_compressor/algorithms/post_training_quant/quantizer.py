# Copyright (c) 2023 Intel Corporation
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
"""Quantizer for onnx models."""

import copy
import logging
import os

import numpy as np
import onnx
import onnxruntime as ort

from onnx_neural_compressor import logger, onnx_model
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant.operators import base_op


class Quantizer:
    """Quantizer class."""

    def __init__(
        self,
        model,
        q_config,
        mode,
        static,
        quantization_params,
        op_types_to_quantize,
        fallback_list=["fp32"],
        reduce_range=None,
        add_qdq_pair_to_weight=False,
        optypes_to_exclude_output_quant=[],
        dedicated_qdq_pair=False,
        execution_provider="CPUExecutionProvider",
    ):
        """Initialization.

        Args:
            model (ModelProto or onnx_model.ONNXModel): onnx model or onnx model wrapper by neural compressor
            q_config (dict): op-wise quantization config.
            mode (str): quantizaion mode
            static (bool): static or not
            quantization_params (dict): scale and zero point of tensors
            op_types_to_quantize (list): optypes to quantize
            fallback_list (list, optional): fallback data type. Defaults to ['fp32'].
            reduce_range (bool, optional): use 7 bit or not. Defaults to None.
            add_qdq_pair_to_weight (bool, optional): add QDQ pair to weight or not. Defaults to False.
            optypes_to_exclude_output_quant (list, optional): optypes to exclude output quantization. Defaults to [].
            dedicated_qdq_pair (bool, optional): dedicate QDQ pair or not. Defaults to False.
            execution_provider (str, optional): execution_provider of onnxrt adaptor. Defaults to CPUExecutionProvider
        """
        self.model = onnx_model.ONNXModel(model) if not isinstance(model, onnx_model.ONNXModel) else model
        model = (
            onnx.shape_inference.infer_shapes(self.model.model) if not self.model.is_large_model else self.model.model
        )
        self.config = q_config
        self.execution_provider = execution_provider
        self.reduce_range = reduce_range
        self.mode = mode
        self.quant_format = None
        self.static = static  # use static quantization for inputs.
        self.fuse_dynamic_quant = False
        self.quantization_params = quantization_params
        self.op_types_to_quantize = op_types_to_quantize
        self.fallback_list = fallback_list
        self.new_nodes = []

        self.opset_version = self.check_opset_version()
        self.value_infos = {vi.name: vi for vi in model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in model.graph.output})
        self.value_infos.update({it.name: it for it in model.graph.input})
        self.replace_input = []
        self.remove_nodes = []
        # List of quantized weights
        self.quantized_value_map = {}
        self.new_value_info = {}

        # List of recalculated quantize weight for Gather op.
        self.recalculate_quantized_value = []

        # QuantizeRange tensor name and zero tensor name for scale and zero point calculation.
        # Used when static is False
        self.fixed_qrange_uint8_name = "fixed_quantization_range_uint8"
        self.fixed_qrange_int8_name = "fixed_quantization_range_int8"
        # For uint8 data-type, to compute zero point, we subtract rmin from 0 (represented by fixed_zero_name tensor)
        self.fixed_zero_name = "fixed_zero"
        # For int8 data-type, zero point is always zero (represented by fixed_zero_point_name tensor)
        self.fixed_zero_zp_name = "fixed_zero_zp"

        if not self.static:
            self.op_types_to_exclude_output_quantization = op_types_to_quantize
        else:
            self.op_types_to_exclude_output_quantization = optypes_to_exclude_output_quant

        self.add_qdq_pair_to_weight = add_qdq_pair_to_weight
        self.dedicated_qdq_pair = dedicated_qdq_pair

    def check_opset_version(self):
        """Check opset version."""
        ai_onnx_domain = [
            opset for opset in self.model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
        ]
        if 1 != len(ai_onnx_domain):
            raise ValueError("Failed to find proper ai.onnx domain")
        opset_version = ai_onnx_domain[0].version

        if opset_version > 10:
            self.fuse_dynamic_quant = True
        elif opset_version < 10:
            logger.warning(
                f"Warning: The original model opset version is {opset_version}, which does not support node "
                + "fusions. Please update the model to opset >= 11 for better performance."
            )
            self.model.model.opset_import.remove(ai_onnx_domain[0])
            self.model.model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
            opset_version = 11

        return opset_version

    def should_quantize(self, node):
        """Check if node should be quantized."""
        if node.name in self.config and self.config[node.name] not in self.fallback_list:
            return True
        elif (
            quant_utils.get_node_original_name(node) in self.config
            and self.config[quant_utils.get_node_original_name(node)] not in self.fallback_list
        ):
            return True
        else:
            return False

    def should_convert(self, node):
        """Check if node should be converted."""
        name = quant_utils.get_node_original_name(node)
        if name in self.config and self.config[name] not in self.fallback_list:
            return True
        else:
            return False

    def _postprocess(self):
        if "TensorrtExecutionProvider" in self.execution_provider:
            quant_utils.trt_env_setup(self.model.model)
        self.merge_dedicated_qdq_pair()
        self.model.remove_unused_nodes()

        self.model.model.producer_name = quant_utils.__producer__
        self.model.model.producer_version = quant_utils.__version__

    def quantize_model(self):
        """Quantize onnx model."""
        # step 1: insert q-dq pairs
        self.insert_qdq()
        self.remove_duplicate_qdq_paris()

        # step 2: convert q-node-dq to qoperator format if needed
        if self.quant_format != "qdq":
            self.convert_qdq_to_operator_oriented()

        self._postprocess()
        return self.model.model

    def merge_dedicated_qdq_pair(self):
        """Merge dedicated Q/DQ pairs."""
        self.remove_nodes = []
        self.replace_input = []
        self.new_nodes = []
        if self.quant_format == "qdq" and self.dedicated_qdq_pair:
            #    node         node
            #     |           / \
            #     q     ->   q   q
            #    / \        /     \
            #  dq   dq    dq       dq
            for node in self.model.nodes():
                if node.op_type in ["QuantizeLinear"]:
                    children = self.model.get_children(node)
                    if len([i for i in children if i.op_type in ["DequantizeLinear"]]) < 2:
                        continue
                    for idx, child in enumerate(children):
                        if child.op_type not in ["DequantizeLinear"]:
                            continue
                        if self.should_quantize(self.model.get_children(child)[0]):
                            inputs = [self.model.get_parents(node)[0].output[0], node.input[1], node.input[2]]
                            self.new_nodes.append(
                                onnx.helper.make_node(
                                    "QuantizeLinear",
                                    inputs,
                                    [node.output[0] + "_" + str(idx)],
                                    node.name + "_" + str(idx),
                                )
                            )
                            self.replace_input.append([child, node.output[0], node.output[0] + "_" + str(idx)])
                        else:
                            self.remove_nodes.append(child)
                            self.replace_input.append(
                                [self.model.get_children(child)[0], child.output[0], node.input[0]]
                            )
                    self.remove_nodes.append(node)
            self.model.remove_nodes(self.remove_nodes)
            self.model.graph().node.extend(self.new_nodes)
            for node, old_input_name, new_input_name in self.replace_input:
                self.model.replace_node_input(node, old_input_name, new_input_name)
            self.model.update()

        elif self.quant_format != "qdq" or not self.dedicated_qdq_pair:
            #      node            node
            #      /  \      ->     |
            #   q(dq)  q(dq)      q(dq)
            target_type = ["QuantizeLinear", "DequantizeLinear"]
            for op_type in target_type:
                for node in self.model.nodes():
                    children = self.model.get_children(node)
                    dq_nodes = [i for i in children if i.op_type == op_type]
                    if len(dq_nodes) < 2 or node.op_type in ["Split"]:
                        continue
                    datas = []
                    for n in dq_nodes:
                        datas.append(
                            [
                                onnx.numpy_helper.to_array(
                                    quant_utils.find_by_name(n.input[1], self.model.initializer())
                                ),
                                onnx.numpy_helper.to_array(
                                    quant_utils.find_by_name(n.input[2], self.model.initializer())
                                ),
                            ]
                        )
                    for idx, data in enumerate(datas):
                        repeaded_id = [i for i, item in enumerate(datas[idx:]) if item == data]
                        for i in repeaded_id[1:]:
                            self.remove_nodes.append(dq_nodes[i])
                            self.replace_input.append(
                                [
                                    self.model.get_children(dq_nodes[i])[0],
                                    dq_nodes[i].output[0],
                                    dq_nodes[idx].output[0],
                                ]
                            )
                self.model.remove_nodes(self.remove_nodes)
                self.model.graph().node.extend(self.new_nodes)
                for node, old_input_name, new_input_name in self.replace_input:
                    self.model.replace_node_input(node, old_input_name, new_input_name)
                self.model.update()

        if self.quant_format == "qdq":
            #       node          node
            #     /  |  \          |
            #    A   q   B   ->    q
            #        |             |
            #        dq            dq
            #                    /   \
            #                   A     B
            for node in self.model.nodes():
                if node.op_type in ["QuantizeLinear"] and len(self.model.get_parents(node)) > 0:
                    if "QuantizeLinear" in [sibling.op_type for sibling in self.model.get_siblings(node)]:
                        continue
                    for sibling in self.model.get_siblings(node):
                        if not self.should_quantize(sibling) and sibling.op_type in base_op.OPERATORS[self.mode]:
                            for inp_idx in range(len(sibling.input)):
                                if sibling.input[inp_idx] == node.input[0]:
                                    self.replace_input.append(
                                        [sibling, sibling.input[inp_idx], self.model.get_children(node)[0].output[0]]
                                    )
            for node, old_input_name, new_input_name in self.replace_input:
                self.model.replace_node_input(node, old_input_name, new_input_name)
            self.model.update()

    def remove_duplicate_qdq_paris(self):
        """Remove duplicated qdq pairs."""
        self.remove_nodes = []
        for node in self.model.nodes():
            if node.op_type == "DequantizeLinear":
                matched_parents = self.model.match_parent_path(
                    node,
                    ["QuantizeLinear", "DequantizeLinear", "QuantizeLinear"],
                    [None, None, None],
                )

                if matched_parents is not None:
                    # (node) DQ - (matched_parents) Q-DQ-Q
                    if all(
                        [i.op_type == "QuantizeLinear" for i in self.model.get_children(matched_parents[1])]
                    ) and not self.model.is_graph_output(matched_parents[1].output[0]):
                        self.remove_nodes.append(matched_parents[1])
                    if all([i.op_type == "DequantizeLinear" for i in self.model.get_children(matched_parents[0])]):
                        self.remove_nodes.append(matched_parents[0])
                        self.replace_input.append([node, node.input[0], matched_parents[2].output[0]])

        self.model.remove_nodes(self.remove_nodes)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)

    def insert_qdq(self):
        """Insert Q/DQ pairs."""
        for node in self.model.nodes():
            if self.should_quantize(node):
                op_quantizer = base_op.OPERATORS[self.mode][node.op_type](self, node)
                if op_quantizer.quantize_check():
                    op_quantizer.quantize()
        self.model.graph().node.extend(self.new_nodes)
        self.model.remove_nodes(self.remove_nodes)

        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()

    def convert_qdq_to_operator_oriented(self):
        """Convert QDQ to QOperator format."""
        self.new_nodes = []
        self.remove_nodes = []
        self.replace_input = []
        for node in self.model.nodes():
            if node.op_type not in ["QuantizeLinear", "DequantizeLinear"] and self.should_convert(node):
                op_converter = base_op.OPERATORS[self.mode][node.op_type](self, node)
                if op_converter.convert_check():
                    op_converter.convert()
        self.model.graph().node.extend(self.new_nodes)
        self.model.remove_nodes(self.remove_nodes)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)
        self.model.update()

    def quantize_bias_tensor(self, node):
        """Quantize bias."""
        input_name, weight_name, bias_name = node.input
        if (
            self.quantization_params is None
            or input_name not in self.quantization_params
            or input_name not in self.quantized_value_map
            or (
                input_name in self.quantized_value_map
                and quant_utils.find_by_name(self.quantized_value_map[input_name].scale_name, self.model.initializer())
                is None
            )
        ):
            self._dynamic_quantize_bias(input_name, weight_name + "_scale", bias_name, bias_name + "_quantized")
        else:
            beta = 1.0
            if node.op_type in ["Gemm"]:
                beta_attribute = [attr for attr in node.attribute if attr.name == "beta"]
                if len(beta_attribute):
                    beta = onnx.helper.get_attribute_value(beta_attribute[0])
            _, quant_value = self.quantize_bias(bias_name, input_name, weight_name, beta)
            if self.model.get_initializer_share_num(bias_name) == 1:
                self.model.remove_initializer(quant_utils.find_by_name(bias_name, self.model.initializer()))
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            axis = None
            if quant_utils.find_by_name(weight_name + "_DequantizeLinear", self.new_nodes):
                dq_node = quant_utils.find_by_name(weight_name + "_DequantizeLinear", self.new_nodes)
                if dq_node.op_type == "DequantizeLinear" and quant_utils.find_by_name("axis", dq_node.attribute):
                    axis = quant_utils.find_by_name("axis", dq_node.attribute).i
            dequant_node = onnx.helper.make_node(
                "DequantizeLinear",
                inputs,
                [bias_name + "_dequantized"],
                bias_name + "_DequantizeLinear",
                axis=axis,
            )
            self.new_nodes.append(dequant_node)
            self.replace_input.append(
                [quant_utils.find_by_name(node.name, self.model.nodes()), bias_name, bias_name + "_dequantized"]
            )

    def quantize_bias(self, bias_name, input_name, weight_name, beta=1.0):
        """Quantized the bias.

        Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        """
        # get scale for weight
        weight_scale_initializer = quant_utils.find_by_name(weight_name + "_scale", self.model.initializer())
        weight_scale = (
            self.tensor_proto_to_array(weight_scale_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(weight_scale_initializer)
        )

        # get bias
        bias_initializer = quant_utils.find_by_name(bias_name, self.model.initializer())
        bias_data = (
            self.tensor_proto_to_array(bias_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(bias_initializer)
        )
        quantized_bias_name = bias_name + "_quantized"

        if input_name in self.quantized_value_map:
            input_scale_name = self.quantized_value_map[input_name].scale_name
        elif input_name in self.quantization_params:
            _, input_scale_name, _, _, _ = self._get_quantization_params(input_name)
        else:
            raise ValueError(f"Expected {input_name} to be in quantized value map for static quantization")
        inputscale_initializer = quant_utils.find_by_name(input_scale_name, self.model.initializer())
        input_scale = (
            self.tensor_proto_to_array(inputscale_initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(inputscale_initializer)
        )

        # calculate scale for bias

        bias_scale = input_scale * weight_scale * beta

        # quantize bias
        quantized_data = (np.asarray(bias_data) / bias_scale).round().astype(np.int32)

        # update bias initializer
        bias_np_data = np.asarray(quantized_data, dtype=np.int32).reshape(bias_initializer.dims)
        packed_bias_initializer = onnx.numpy_helper.from_array(bias_np_data, quantized_bias_name)
        self.model.initializer().extend([packed_bias_initializer])

        # update scale initializer
        quantized_bias_scale_name = bias_name + "_scale"
        bias_scale_data = np.asarray(bias_scale, dtype=np.float32).reshape(-1)
        packed_bias_scale_initializer = onnx.numpy_helper.from_array(bias_scale_data, quantized_bias_scale_name)
        self.model.initializer().extend([packed_bias_scale_initializer])

        # update zero initializer
        quantized_bias_zp_name = bias_name + "_zero_point"
        bias_zp_data = np.zeros(bias_scale.shape, dtype=np.int32).reshape(-1)
        packed_bias_zp_initializer = onnx.numpy_helper.from_array(bias_zp_data, quantized_bias_zp_name)
        self.model.initializer().extend([packed_bias_zp_initializer])

        # log entries for this quantized bias value
        quantized_bias_entry = quant_utils.QuantizedInitializer(
            bias_name,
            bias_initializer,
            [0],
            [0],
            [0],
            [bias_scale],
            bias_data,
            quantized_data,
            qType=onnx.TensorProto.INT32,
        )

        quantized_value = quant_utils.QuantizedValue(
            bias_name,
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            quant_utils.QuantizedValueType.Initializer,
            None,
            onnx.TensorProto.INT32,
        )
        return quantized_bias_name, quantized_value

    def quantize_weight_per_channel(self, weight_name, weight_qType, sym, channel_axis):
        """Quantize weight per-channel."""
        name = (
            ("_").join([weight_name, str(weight_qType)])
            if self.model.get_initializer_share_num(weight_name) > 1
            else weight_name
        )
        if name in self.quantized_value_map:
            return (name + "_quantized", name + "_zero_point", name + "_scale")

        initializer = quant_utils.find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = (
            self.tensor_proto_to_array(initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(initializer)
        )
        rmin, rmax, zero_point, scale, quantized_weights = quant_utils.quantize_data_per_channel(
            weights,
            channel_axis,
            quant_utils.get_qmin_qmax_for_qType(weight_qType, self.reduce_range, sym),
            weight_qType,
            sym,
        )

        weight = quant_utils.QuantizedInitializer(
            name,
            initializer,
            rmin,
            rmax,
            zero_point,
            scale,
            weights,
            quantized_weights.flatten().tolist(),
            channel_axis,
            weight_qType,
        )

        self._update_weight(weight)
        quantized_value = quant_utils.QuantizedValue(
            weight.name,
            weight.name + "_quantized",
            weight.name + "_scale",
            weight.name + "_zero_point",
            quant_utils.QuantizedValueType.Initializer,
            None,
            weight_qType,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return (weight.name + "_quantized", weight.name + "_zero_point", weight.name + "_scale")

    def dequantize_tensor(self, node, value_name):
        """Dequantize tensor."""
        if value_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[value_name]
            dqlinear_name = value_name + "_DequantizeLinear"
            dqlinear_inputs = [value_name + "_quantized", quantized_value.scale_name, quantized_value.zp_name]
            dequantize_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [value_name], dqlinear_name)
            if dequantize_node not in self.new_nodes:
                self.new_nodes.append(dequantize_node)
        else:  # pragma: no cover
            data_found, scale_name, zp_name, _, _ = self._get_quantization_params(value_name)
            if self.static:
                if data_found is False:
                    raise ValueError(
                        "Quantization parameters are not specified for param {}."
                        "In static mode quantization params for inputs and outputs "
                        "of nodes to be quantized are required.".format(value_name)
                    )
            dqlinear_name = value_name + "_DequantizeLinear"
            dqlinear_inputs = [value_name + "_quantized", scale_name, zp_name]
            dequantize_node = onnx.helper.make_node("DequantizeLinear", dqlinear_inputs, [value_name], dqlinear_name)
            if dequantize_node not in self.new_nodes:
                self.new_nodes.append(dequantize_node)

    def _update_weight(self, weight):
        """Update weight.

        Given a weight object, update the graph by doing the following:
         - remove old initializer, update new initializers for
           quantized weight, zero point, and scale
         - remove old weight input, update with new inputs for
           quantized weight, zero point, and scale
        This function does NOT update the nodes in the graph, just initializers and inputs
        """
        if weight.name in self.quantized_value_map:
            return
        packed_weight_name = weight.name + "_quantized"
        scale_name = weight.name + "_scale"
        zero_point_name = weight.name + "_zero_point"

        # Update packed weight, zero point, and scale initializers
        packed_weight_np_data = np.asarray(
            weight.quantized_data, dtype=onnx.helper.tensor_dtype_to_np_dtype(weight.qType)
        ).reshape(weight.initializer.dims)
        packed_weight_initializer = onnx.numpy_helper.from_array(packed_weight_np_data, packed_weight_name)

        if not self.add_qdq_pair_to_weight or self.quant_format != "qdq":
            self.model.initializer().append(packed_weight_initializer)
        if weight.axis is not None:
            zero_scale_shape = [weight.initializer.dims[weight.axis]]
        else:  # scale and zero point must be scalar
            zero_scale_shape = []
        zero_point_type = weight.qType
        scale_initializer = onnx.helper.make_tensor(
            scale_name, weight.initializer.data_type, zero_scale_shape, weight.scales
        )
        zero_initializer = onnx.helper.make_tensor(
            zero_point_name, zero_point_type, zero_scale_shape, weight.zero_points
        )

        self.model.initializer().extend([scale_initializer, zero_initializer])

    @staticmethod
    def tensor_proto_to_array(initializer, base_dir=""):
        """Convert TensorProto to array."""
        if quant_utils.is_quantizable_type(initializer.data_type):
            weights = onnx.numpy_helper.to_array(initializer, base_dir)
        else:
            raise ValueError(
                "Only float type quantization is supported. \
                Weights {} is {}.".format(
                    initializer.name, quant_utils.dtype_to_name(quant_utils.dtype_mapping, initializer.data_type)
                )
            )
        return weights

    def _get_quantization_params(self, param_name):
        """Create initializers and inputs in the graph for zero point and scale of output.

        Zero point and scale values are obtained from self.quantization_params if specified.

        Args:
            param_name (string): Name of the quantization parameter.
        """
        if self.quantization_params is None or param_name not in self.quantization_params:
            return False, "", "", "", ""

        params = self.quantization_params[param_name]
        if params is None or len(params) != 2:
            raise ValueError(
                "Quantization parameters should contain zero point and scale. "
                "Specified values for output {}: {}".format(param_name, params)
            )

        zero_point_values = [params[0]]
        zero_point_shape = []
        zero_point_name = param_name + "_zero_point"
        zero_point_type = onnx.helper.np_dtype_to_tensor_dtype(params[0].dtype)

        scale_values = [params[1]]
        scale_shape = []
        scale_name = param_name + "_scale"
        scale_dtype = onnx.helper.np_dtype_to_tensor_dtype(params[1].dtype)

        # Add initializers
        init_zp = onnx.helper.make_tensor(zero_point_name, zero_point_type, zero_point_shape, zero_point_values)
        self.model.add_initializer(init_zp)
        init_scale = onnx.helper.make_tensor(scale_name, scale_dtype, scale_shape, scale_values)
        self.model.add_initializer(init_scale)

        return True, scale_name, zero_point_name, scale_shape, zero_point_shape

    def _get_quantized_weight(self, initializer, qType, sym):
        """Get quantized weight."""
        name = (
            ("_").join([initializer.name, str(qType)])
            if self.model.get_initializer_share_num(initializer.name) > 1
            else initializer.name
        )
        if name in self.quantized_value_map:
            return self.quantized_value_map[name]
        weights_data = (
            self.tensor_proto_to_array(initializer, os.path.dirname(self.model.model_path))
            if self.model.model_path is not None
            else self.tensor_proto_to_array(initializer)
        )
        rmin, rmax, zero_point, scale, quantized_weights_data = quant_utils.quantize_data(
            weights_data.flatten().tolist(),
            quant_utils.get_qmin_qmax_for_qType(qType, self.reduce_range, sym),
            qType,
            sym,
        )
        weight = quant_utils.QuantizedInitializer(
            name,
            initializer,
            [rmin],
            [rmax],
            [zero_point],
            [scale],
            weights_data,
            quantized_weights_data,
            axis=None,
            qType=qType,
        )

        return weight

    def is_valid_quantize_weight(self, weight_name):
        """Check weight can be quantized."""
        weight = quant_utils.find_by_name(weight_name, self.model.initializer())
        if weight is not None:
            return quant_utils.is_quantizable_type(weight.data_type)
        else:
            return weight_name in self.quantized_value_map

    def get_bias_add_nodes(self, node, weight_name, last_output, quantized_bias_name):
        """Given a node, this function handles bias add by adding a "reshape" node on bias and an "add" node.

        Args:
            node (NodeProto): current node (Conv)
            weight_name (string): weight name
            last_output (_type_): output of previous node (input to bias add)
            quantized_bias_name (string): bias name
        """
        # Add tensors for the shape to be reshaped to
        weight = quant_utils.find_by_name(weight_name, self.model.initializer())
        if weight is None:
            raise ValueError("Expected {} to be an initializer".format(node.input[1]))

        # Add reshape for correct broadcast
        reshape_input_data = quantized_bias_name
        reshape_input_shape = quantized_bias_name + "_reshape_shape"
        reshape_input = [reshape_input_data, reshape_input_shape]
        reshape_shape = np.ones((len(weight.dims)), dtype=np.int64)
        reshape_shape[1] = -1
        init_shape = onnx.helper.make_tensor(
            reshape_input_shape, onnx.TensorProto.INT64, [len(weight.dims)], reshape_shape
        )
        self.model.add_initializer(init_shape)

        reshape_op_output = node.output[0] + "_reshape"
        reshape_node = onnx.helper.make_node(
            "Reshape", reshape_input, [reshape_op_output], quantized_bias_name + "reshape"
        )
        self.new_nodes.append(reshape_node)

        # Add an Add operation for bias
        bias_add_input = [last_output]
        bias_add_input.append(reshape_op_output)
        add_node_output = node.output[0] + "_bias_add"
        add_node = onnx.helper.make_node("Add", bias_add_input, [add_node_output], quantized_bias_name + "bias_add")
        self.new_nodes.append(add_node)
        return add_node_output

    def quantize_outputs(self, node, initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node outputs."""
        for idx, tensor_name in enumerate(node.output):
            if (
                tensor_name in self.value_infos
                and self.value_infos[tensor_name].type.HasField("tensor_type")
                and not quant_utils.is_quantizable_type(self.value_infos[tensor_name].type.tensor_type.elem_type)
            ):
                return
            data_found = False
            refer_name = node.input[0] if direct_int8 else tensor_name

            if refer_name in self.quantized_value_map:
                scale_name = self.quantized_value_map[refer_name].scale_name
                zp_name = self.quantized_value_map[refer_name].zp_name
                data_found = True
            elif refer_name in self.quantization_params:
                data_found, scale_name, zp_name, _, _ = self._get_quantization_params(refer_name)

            if data_found is False:
                raise ValueError(
                    "Quantization parameters are not specified for param {}."
                    "In static mode quantization params for inputs and outputs "
                    "of nodes to be quantized are required.".format(tensor_name)
                )

            node.output[idx] = tensor_name + "_QuantizeInput"
            q_input = node.output[idx]
            q_output = tensor_name + "_quantized"
            dq_input = q_output
            dq_output = tensor_name
            quant_node_name = tensor_name + "_" + node.name + "_QuantizeLinear"
            dequant_node_name = tensor_name + "_" + node.name + "_DequantizeLinear"
            qlinear_node = onnx.helper.make_node(
                "QuantizeLinear",
                [q_input, scale_name, zp_name],
                [q_output],
                quant_node_name,
            )
            dequant_node = onnx.helper.make_node(
                "DequantizeLinear",
                [dq_input, scale_name, zp_name],
                [dq_output],
                dequant_node_name,
            )
            self.new_nodes.extend([qlinear_node, dequant_node])
            for child in self.model.get_children(node):
                self.replace_input.append([child, tensor_name, dequant_node.output[0]])
            if tensor_name not in self.quantized_value_map:
                quantized_value = quant_utils.QuantizedValue(
                    tensor_name, dq_output, scale_name, zp_name, quant_utils.QuantizedValueType.Input
                )
                self.quantized_value_map[tensor_name] = quantized_value

    def quantize_inputs(self, node, indices=None, initializer_use_weight_qType=True, direct_int8=False):
        """Quantize node inputs."""
        # Quantize the input
        for idx, tensor_name in enumerate(node.input):
            if indices and idx not in indices:
                continue
            initializer = quant_utils.find_by_name(tensor_name, self.model.initializer())
            if initializer is not None:
                if not quant_utils.is_quantizable_type(initializer.data_type):
                    return

                dtype = (
                    self.config[node.name]["weight_type"]
                    if initializer_use_weight_qType
                    else self.config[node.name]["activation_type"]
                )
                sym = (
                    self.config[node.name]["weight_sym"]
                    if initializer_use_weight_qType
                    else self.config[node.name]["activation_sym"]
                )
                weight = self._get_quantized_weight(initializer, dtype, sym)
                self._update_weight(weight)
                node.input[idx] = weight.name
                q_weight_name = weight.name + "_quantized"
                zp_name = weight.name + "_zero_point"
                scale_name = weight.name + "_scale"

                if self.add_qdq_pair_to_weight and self.quant_format == "qdq":
                    qlinear_node = onnx.helper.make_node(
                        "QuantizeLinear",
                        [tensor_name, scale_name, zp_name],
                        [weight.name + "_quantized"],
                        weight.name + "_QuantizeLinear",
                    )
                    self.new_nodes.append(qlinear_node)

                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    [q_weight_name, scale_name, zp_name],
                    [weight.name + "_dequantized"],
                    weight.name + "_DequantizeLinear",
                )
                self.new_nodes.append(dequant_node)
                self.replace_input.append([node, weight.name, dequant_node.output[0]])
                if weight.name not in self.quantized_value_map:
                    quantized_value = quant_utils.QuantizedValue(
                        weight.name,
                        q_weight_name,
                        scale_name,
                        zp_name,
                        quant_utils.QuantizedValueType.Initializer,
                        None,
                        dtype,
                    )
                    self.quantized_value_map[weight.name] = quantized_value
            else:
                if (
                    tensor_name in self.value_infos
                    and self.value_infos[tensor_name].type.HasField("tensor_type")
                    and not quant_utils.is_quantizable_type(self.value_infos[tensor_name].type.tensor_type.elem_type)
                ):
                    return
                self._quantize_activation(node, tensor_name, direct_int8)

    def quantize_weights_per_channel(self, node, indices, weight_qType, sym, axis):
        """Quantize weights per-channel."""
        if self.opset_version < 13 and self.quant_format == "qdq":
            self.quantize_inputs(node, indices)
            return

        for idx, inp in enumerate(node.input):
            if idx not in indices:
                continue

            q_name, zp_name, scale_name = self.quantize_weight_per_channel(inp, weight_qType, sym, axis)
            weight_name = ("_").join([inp, str(weight_qType)]) if self.model.get_initializer_share_num(inp) > 1 else inp
            dequant_node = onnx.helper.make_node(
                "DequantizeLinear",
                [q_name, scale_name, zp_name],
                [weight_name + "_dequantized"],
                weight_name + "_DequantizeLinear",
                axis=axis,
            )
            self.new_nodes.append(dequant_node)
            node.input[idx] = weight_name

            # Replace weight_name with output of DequantizeLinear
            self.replace_input.append([node, weight_name, dequant_node.output[0]])

            if self.add_qdq_pair_to_weight and self.quant_format == "qdq":
                qlinear_node = onnx.helper.make_node(
                    "QuantizeLinear",
                    [inp, scale_name, zp_name],
                    [q_name],
                    weight_name + "_QuantizeLinear",
                    axis=axis,
                )
                self.new_nodes.append(qlinear_node)


class StaticQuantizer(Quantizer):
    """Static quantizer class."""

    def __init__(
        self,
        model,
        q_config,
        quant_format="qoperator",
        quantization_params={},
        op_types_to_quantize=[],
        fallback_list=["fp32"],
        reduce_range=None,
        add_qdq_pair_to_weight=False,
        optypes_to_exclude_output_quant=[],
        dedicated_qdq_pair=False,
        execution_provider="CPUExecutionProvider",
    ):
        """Initialization.

        Args:
            model (ModelProto or ONNXModel): onnx model or onnx model wrapper by neural compressor
            q_config (dict): op-wise quantization config.
            static (bool): static or not
            quantization_params (dict): scale and zero point of tensors
            op_types_to_quantize (list): optypes to quantize
            fallback_list (list, optional): fallback data type. Defaults to ['fp32'].
            reduce_range (bool, optional): use 7 bit or not. Defaults to None.
            add_qdq_pair_to_weight (bool, optional): add QDQ pair to weight or not. Defaults to False.
            optypes_to_exclude_output_quant (list, optional): optypes to exclude output quantization. Defaults to [].
            dedicated_qdq_pair (bool, optional): dedicate QDQ pair or not. Defaults to False.
            execution_provider (str, optional): execution_provider of onnxrt adaptor. Defaults to CPUExecutionProvider
        """
        super().__init__(
            mode="static_quant",
            model=model,
            q_config=q_config,
            static=True,
            quantization_params=quantization_params,
            op_types_to_quantize=op_types_to_quantize,
        )
        self.fallback_list = fallback_list
        self.reduce_range = reduce_range
        self.add_qdq_pair_to_weight = add_qdq_pair_to_weight
        self.optypes_to_exclude_output_quant = optypes_to_exclude_output_quant
        self.dedicated_qdq_pair = dedicated_qdq_pair
        self.execution_provider = execution_provider
        self.static = True  # use static quantization for inputs.
        self.quant_format = quant_format
        if self.opset_version < 13 and self.quant_format == "qdq":
            logger.warning(
                "Per-channel support with QDQ format requires opset version >= 13,"
                " use per-tensor granularity instead"
            )
        if "TensorrtExecutionProvider" in execution_provider:

            # TensorrtExecutionProvider doesn't support Conv + Add fusion
            self._revert_conv_add_fusion()

            # only quantize Add which is followed by ReduceMean
            for node in self.model.nodes():
                if node.op_type == "Add":
                    children = self.model.get_children(node)
                    if "ReduceMean" not in [i.op_type for i in children]:
                        self.config[node.name] = "fp32"

    def _revert_conv_add_fusion(self):
        add_nodes = []
        remove_nodes = []
        for node in self.model.nodes():
            if node.op_type == "Conv" and len(node.input) == 3:
                bias_tensor = self.model.get_initializer(node.input[2])
                bias_array = onnx.numpy_helper.to_array(bias_tensor).reshape((-1, 1, 1))
                self.model.remove_initializer(bias_tensor)
                self.model.add_initializer(onnx.numpy_helper.from_array(bias_array, bias_tensor.name))
                kwargs = {}
                activation_params = None
                for attr in node.attribute:
                    kwargs.update(quant_utils.attribute_to_kwarg(attr))
                conv = onnx.helper.make_node("Conv", node.input[0:2], [node.name + "_revert"], node.name, **kwargs)
                add = onnx.helper.make_node("Add", [conv.output[0], node.input[2]], node.output, node.name + "_add")
                add_nodes.extend([conv, add])

        self.model.remove_nodes(remove_nodes)
        self.model.add_nodes(add_nodes)
        self.model.update()

    def _quantize_activation(self, node, tensor_name, direct_int8=False):
        """Quantize node activation."""
        if tensor_name in self.quantized_value_map:
            scale_name = self.quantized_value_map[tensor_name].scale_name
            zp_name = self.quantized_value_map[tensor_name].zp_name
            data_found = True
        else:
            data_found, scale_name, zp_name, _, _ = self._get_quantization_params(tensor_name)

        if data_found is False:
            raise ValueError(
                "Quantization parameters are not specified for param {}."
                "In static mode quantization params for inputs and outputs "
                "of nodes to be quantized are required.".format(tensor_name)
            )

        if direct_int8:
            # direct int8 models will be quantized only if their inputs are quantized
            if node.input[0] not in self.quantized_value_map:
                return

        q_input = tensor_name
        q_output = (
            tensor_name + "_" + node.name + "_QuantizeLinear"
            if tensor_name not in self.model.input()
            else tensor_name + "_quantized"
        )
        dq_input = q_output
        dq_output = (
            tensor_name + "_" + node.name + "_dequantized"
            if tensor_name not in self.model.input()
            else tensor_name + "_dequantized"
        )
        self.replace_input.append([node, tensor_name, dq_output])

        if tensor_name in self.model.input() and tensor_name in self.quantized_value_map:
            return

        quant_node_name = tensor_name + "_" + node.name + "_QuantizeLinear"
        dequant_node_name = tensor_name + "_" + node.name + "_DequantizeLinear"
        qlinear_node = onnx.helper.make_node(
            "QuantizeLinear",
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
        )
        dequant_node = onnx.helper.make_node(
            "DequantizeLinear",
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
        )
        self.new_nodes.extend([qlinear_node, dequant_node])

        if tensor_name not in self.quantized_value_map:
            quantized_value = quant_utils.QuantizedValue(
                tensor_name, dq_output, scale_name, zp_name, quant_utils.QuantizedValueType.Input
            )
            self.quantized_value_map[tensor_name] = quantized_value


class DynamicQuantizer(Quantizer):
    """Dynamic quantizer class."""

    def __init__(
        self,
        model,
        q_config,
        quantization_params={},
        op_types_to_quantize=[],
        fallback_list=["fp32"],
        reduce_range=None,
        execution_provider="CPUExecutionProvider",
    ):
        """Initialization.

        Args:
            model (ModelProto or onnx_model.ONNXModel): onnx model or onnx model wrapper by neural compressor
            q_config (dict): op-wise quantization config.
            quantization_params (dict): scale and zero point of tensors
            op_types_to_quantize (list): optypes to quantize
            fallback_list (list, optional): fallback data type. Defaults to ['fp32'].
            reduce_range (bool, optional): use 7 bit or not. Defaults to None.
            add_qdq_pair_to_weight (bool, optional): add QDQ pair to weight or not. Defaults to False.
            dedicated_qdq_pair (bool, optional): dedicate QDQ pair or not. Defaults to False.
            execution_provider (str, optional): execution_provider of onnxrt adaptor. Defaults to CPUExecutionProvider
        """
        super().__init__(
            mode="dynamic_quant",
            model=model,
            q_config=q_config,
            static=False,
            quantization_params=quantization_params,
            op_types_to_quantize=op_types_to_quantize,
        )

    def _quantize_activation(self, node, tensor_name, direct_int8=False):
        """Quantize node activation."""
        qlinear_node = self.model.find_node_by_name(tensor_name + "_QuantizeLinear", self.new_nodes, self.model.graph())
        if qlinear_node is None:
            if (
                self.fuse_dynamic_quant
                and self.config[node.name]["activation_type"] == onnx.TensorProto.UINT8
                and not self.config[node.name]["activation_sym"]
            ):
                # DynamicQuantizeLinear supports uint8 input for CPU EP, supports uint8 and int8 for DML EP
                scale_name = tensor_name + "_scale"
                zp_name = tensor_name + "_zero_point"
                if quant_utils.find_by_name(scale_name, self.model.initializer()):
                    self.model.remove_initializer(quant_utils.find_by_name(scale_name, self.model.initializer()))
                if quant_utils.find_by_name(zp_name, self.model.initializer()):
                    self.model.remove_initializer(quant_utils.find_by_name(zp_name, self.model.initializer()))
                qlinear_node = onnx.helper.make_node(
                    "DynamicQuantizeLinear",
                    [tensor_name],
                    [tensor_name + "_dynamic_quantized", scale_name, zp_name],
                    tensor_name + "_QuantizeLinear",
                )
            else:
                scale_name, zp_name, _, _ = self._get_dynamic_input_quantization_params(
                    tensor_name, self.config[node.name]["activation_type"]
                )
                qlinear_node = onnx.helper.make_node(
                    "QuantizeLinear",
                    [tensor_name, scale_name, zp_name],
                    [tensor_name + "_quantized"],
                    tensor_name + "_QuantizeLinear",
                )
            if qlinear_node not in self.new_nodes:
                self.new_nodes.append(qlinear_node)
            self.quantized_value_map[tensor_name] = quant_utils.QuantizedValue(
                tensor_name,
                qlinear_node.output[0],
                scale_name,
                zp_name,
                self.config[node.name]["activation_type"],
            )
        self.replace_input.append([node, tensor_name, qlinear_node.output[0]])

    def _get_dynamic_input_quantization_params(self, input_name, qType):
        """Create nodes for dynamic quantization of input.

        Args:
            input_name (string): Name of the input.
            qType (int): type to quantize to.
        """
        if qType == onnx.TensorProto.INT8:
            return self._get_dynamic_input_quantization_params_int8(input_name)

        return self._get_dynamic_input_quantization_params_uint8(input_name)

    def _get_dynamic_input_quantization_params_int8(self, input_name):  # pragma: no cover
        """Create nodes for dynamic quantization of input to int8.

        Args:
            input_name (string): Name of the input.
        """
        qType = onnx.TensorProto.INT8

        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_max_node)

        # Compute scale
        #   Find abs(rmin)
        reduce_min_abs_name = reduce_min_name + "_Abs"
        reduce_min_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_min_node.output[0]],
            [reduce_min_abs_name + ":0"],
            reduce_min_abs_name,
        )
        self.new_nodes.append(reduce_min_abs_node)
        #   Find abs(rmax)
        reduce_max_abs_name = reduce_max_name + "_Abs"
        reduce_max_abs_node = onnx.helper.make_node(
            "Abs",
            [reduce_max_node.output[0]],
            [reduce_max_abs_name + ":0"],
            reduce_max_abs_name,
        )
        self.new_nodes.append(reduce_max_abs_node)
        #   Compute max of abs(rmin) and abs(rmax)
        abs_max_name = input_name + "_Abs_Max"
        abs_max_node = onnx.helper.make_node(
            "Max",
            [reduce_min_abs_node.output[0], reduce_max_abs_node.output[0]],
            [abs_max_name + ":0"],
            abs_max_name,
        )
        self.new_nodes.append(abs_max_node)
        #   and divide by (quantize_range/2.0) which will be equal to max(...)*2.0/quantize_range
        qmin, qmax = quant_utils.get_qmin_qmax_for_qType(qType, self.reduce_range)
        initializer_div = onnx.helper.make_tensor(
            self.fixed_qrange_int8_name,
            onnx.TensorProto.FLOAT,
            [],
            [(qmax - qmin) / 2.0],
        )
        self.model.add_initializer(initializer_div)
        scale_div_name = input_name + "scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [abs_max_node.output[0], self.fixed_qrange_int8_name],
            [input_scale_name],
            scale_div_name,
        )
        self.new_nodes.append(scale_div_node)

        # Zero point
        initializer_zp = onnx.helper.make_tensor(self.fixed_zero_zp_name, qType, [], [0])
        self.model.add_initializer(initializer_zp)

        return input_scale_name, self.fixed_zero_zp_name, [], []

    def _get_dynamic_input_quantization_params_uint8(self, input_name):
        """Create nodes for dynamic quantization of input to uint8.

        Args:
            input_name (string): Name of the input.
        """
        qType = onnx.TensorProto.UINT8
        # Reduce min and Reduce max
        input_scale_name = input_name + "_scale"
        input_zp_name = input_name + "_zero_point"

        reduce_min_name = input_name + "_ReduceMin"
        reduce_min_node = onnx.helper.make_node(
            "ReduceMin",
            [input_name],
            [reduce_min_name + ":0"],
            reduce_min_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_min_node)

        reduce_max_name = input_name + "_ReduceMax"
        reduce_max_node = onnx.helper.make_node(
            "ReduceMax",
            [input_name],
            [reduce_max_name + ":0"],
            reduce_max_name,
            keepdims=0,
        )
        self.new_nodes.append(reduce_max_node)

        # Add tensors for quantize range and zero value.
        qmin, qmax = quant_utils.get_qmin_qmax_for_qType(qType, self.reduce_range)
        initializer_qrange = onnx.helper.make_tensor(
            self.fixed_qrange_uint8_name,
            onnx.TensorProto.FLOAT,
            [],
            [qmax - qmin],
        )
        self.model.add_initializer(initializer_qrange)
        initializer_qvalue = onnx.helper.make_tensor(self.fixed_zero_name, onnx.TensorProto.FLOAT, [], [0.0])
        self.model.add_initializer(initializer_qvalue)

        # Compute Scale
        #   Subtract rmax and rmin
        scale_sub_name = input_name + "_scale_Sub"
        scale_sub_node = onnx.helper.make_node(
            "Sub",
            [reduce_max_node.output[0], reduce_min_node.output[0]],
            [scale_sub_name + ":0"],
            scale_sub_name,
        )
        self.new_nodes.append(scale_sub_node)
        #   and divide by quantize range
        scale_div_name = input_name + "_scale_Div"
        scale_div_node = onnx.helper.make_node(
            "Div",
            [scale_sub_node.output[0], self.fixed_qrange_uint8_name],
            [input_scale_name],
            scale_div_name,
        )
        self.new_nodes.append(scale_div_node)

        # Compute zero point
        #   Subtract zero and rmin
        zp_sub_name = input_name + "_zero_point_Sub"
        zp_sub_node = onnx.helper.make_node(
            "Sub",
            [self.fixed_zero_name, reduce_min_node.output[0]],
            [zp_sub_name + ":0"],
            zp_sub_name,
        )
        self.new_nodes.append(zp_sub_node)
        #   Divide by scale
        zp_div_name = input_name + "_zero_point_Div"
        zp_div_node = onnx.helper.make_node(
            "Div",
            [zp_sub_node.output[0], input_scale_name],
            [zp_div_name + ":0"],
            zp_div_name,
        )
        self.new_nodes.append(zp_div_node)
        #   Compute floor
        zp_floor_name = input_name + "_zero_point_Floor"
        zp_floor_node = onnx.helper.make_node("Floor", zp_div_node.output, [zp_floor_name + ":0"], zp_floor_name)
        self.new_nodes.append(zp_floor_node)
        #   Cast to integer
        zp_cast_name = input_name + "_zero_point_Cast"
        zp_cast_node = onnx.helper.make_node("Cast", zp_floor_node.output, [input_zp_name], zp_cast_name, to=qType)
        self.new_nodes.append(zp_cast_node)

        return input_scale_name, input_zp_name, [], []
