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
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Calibration for onnx models."""

import copy
import logging
import os
import sys
from importlib import util

import numpy as np
import onnx
import onnxruntime
from packaging import version

from onnx_neural_compressor import logger, onnx_model
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant import calibrator

if sys.version_info < (3, 11) and util.find_spec("onnxruntime_extensions"):
    import onnxruntime_extensions

ONNX18_VERSION = version.Version("1.8.0")
ORT112_VERSION = version.Version("1.12.0")


class ONNXRTAugment:
    """Augment input model to dump tensor or for calibration."""

    def __init__(
        self,
        model_wrapper,
        dataloader,
        dump_op_types,
        black_nodes=[],
        white_nodes=[],
        iterations=[],
        execution_provider="CPUExecutionProvider",
        reduce_range=False,
        **kwargs,
    ):
        """Initialization.

        Args:
            model_wrapper (Model): model to be augmented
            dataloader (object): user implemented object to read in and preprocess calibration dataset
            dump_op_types (list): operator types to be calibrated and quantized
            black_nodes (list, optional): operator names that should not be quantized. Defaults to [].
            white_nodes (list, optional): operator names that force to be quantized. Defaults to [].
            iterations (list, optional): tensor of which iteration will be collected. Defaults to [].
            execution_provider (list, optional): execution provider for onnxruntime. Defaults to 'CPUExecutionProvider'.
            reduce_range (bool, optional): use 7 bit or not. Defaults to False.
        """
        self.model_wrapper = (
            model_wrapper
            if isinstance(model_wrapper, onnx_model.ONNXModel)
            else onnx_model.ONNXModel(model_wrapper, load_external_data=True)
        )
        self.model = self.model_wrapper.model
        ai_onnx_domain = [opset for opset in self.model.opset_import if not opset.domain or opset.domain == "ai.onnx"]
        self.opset_version = ai_onnx_domain[0].version
        self.dataloader = dataloader
        self.dump_op_types = dump_op_types
        self.black_nodes = black_nodes
        self.white_nodes = white_nodes
        self.augmented_model = None
        self.iterations = iterations
        self.execution_provider = execution_provider
        self.augment_nodes = []
        self.dequantized_output = {}
        self.already_quantized = "DequantizeLinear" in [node.op_type for node in self.model.graph.node]
        self.dynamically_quantized = False
        self.ort_version = version.Version(onnxruntime.__version__)
        self.reduce_range = reduce_range

    def augment_graph(self):
        """Augment_graph.

        Adds nodes to all quantization_candidates op type nodes in model and
        ensures their outputs are stored as part of the graph output.

        Args:
            activation_only (bool, optional): whether to dump activation tensor only. Defaults to False.
            weight_only (bool, optional): whether to dump weight_only. Defaults to False.
        """
        self.dequantized_output.clear()
        onnx_version = version.Version(onnx.__version__)
        if onnx_version < ONNX18_VERSION:
            logger.warning("Static quantization for NLP model is supported at onnx 1.8.0 and newer.")
        if self.already_quantized and any(
            [i.dims in [1, 2] for i in self.model_wrapper.initializer() if i.name.endswith("_scale")]
        ):
            if self.opset_version < 13 and self.ort_version >= ORT112_VERSION:
                logger.warning(
                    "Please use onnxruntime < 1.12.0 or upgrade model opset "
                    "version to 13 or higher to inspect per-channel quantized weight"
                )

        model = copy.deepcopy(self.model)
        model_nodes_names = [node.name for node in model.graph.node]

        added_nodes = []
        added_outputs = []
        tensors_to_dump = set()

        for augment_node_type in self.augment_nodes:
            if augment_node_type not in ["DequantizeLinear"]:  # pragma: no cover
                raise ValueError(
                    "Unexpected augment_node {} only DequantizeLinear is supported".format(augment_node_type)
                )

        if self.already_quantized:
            # mapping between fp32 node and int8 node
            new_white_nodes = []
            for white_node in self.white_nodes:
                new_white_node = white_node + "_quant"
                assert new_white_node in model_nodes_names, "no quantized {} in the graph".format(white_node)
                new_white_nodes.append(new_white_node)
            self.white_nodes = new_white_nodes

        node_outputs = []
        for node in model.graph.node:  # pylint: disable=no-member
            node_outputs.extend(node.output)
            should_be_dump = ((node.op_type in self.dump_op_types) and (node.name not in self.black_nodes)) or (
                node.name in self.white_nodes
            )
            if should_be_dump:
                # add input tensors which should be dump
                for input in node.input:
                    if len(input) != 0:  # to prevent input is ""
                        initializer_tensor = self.model_wrapper.get_initializer(input)
                        if initializer_tensor is None:
                            tensors_to_dump.add(input)
                # add output tensors which should be dump
                tensors_to_dump.update([output for output in node.output if len(output) != 0])

        model_inputs = [i.name for i in model.graph.input]
        for tensor in tensors_to_dump:
            if tensor not in node_outputs and tensor not in model_inputs:
                continue
            if self.augment_nodes:
                for augment_node_type in self.augment_nodes:
                    if augment_node_type in ["DequantizeLinear"]:
                        # insert DequantizeLinear node as output
                        if tensor.endswith("_scale") or tensor.endswith("_zero_point"):  # pragma: no cover
                            continue

                        if not self.dynamically_quantized:
                            tensor = (
                                tensor.replace("_QuantizeInput", "_quantized")
                                if tensor.endswith("_QuantizeInput")
                                else tensor
                            )
                        else:
                            tensor = (
                                tensor.replace("_output_quantized", "")
                                if tensor.endswith("_output_quantized")
                                else tensor
                            )

                        augment_node_name = tensor + "_new_" + augment_node_type
                        scale, zero_point = self.model_wrapper.get_scale_zero(tensor)
                        if scale:
                            # the tensor is in INT8 dtype
                            nodes, output = self._dequantize(tensor, scale, zero_point)
                            if output:
                                added_nodes.extend(nodes)
                                added_outputs.append(
                                    onnx.helper.make_tensor_value_info(
                                        output, onnx.TensorProto.FLOAT, ()  # pylint: disable=no-member
                                    )
                                )  # pylint: disable=no-member
                        else:
                            # the tensor is in FP32 dtype
                            if tensor not in [t.name for t in model.graph.output]:
                                added_tensor = onnx.helper.ValueInfoProto()
                                added_tensor.name = tensor
                                added_outputs.append(added_tensor)
            else:
                if tensor not in [t.name for t in model.graph.output]:
                    added_tensor = onnx.helper.ValueInfoProto()
                    added_tensor.name = tensor
                    added_outputs.append(added_tensor)

        if self.augment_nodes:
            model.graph.node.extend(added_nodes)  # pylint: disable=no-member
        model.graph.output.extend(added_outputs)  # pylint: disable=no-member

        self.augmented_model = model
        if self.model_wrapper.is_large_model:  # pragma: no cover
            onnx.save_model(
                model,
                self.model_wrapper.model_path + "_augment.onnx",
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )

    def get_activation_tensors_calib_range(self, q_config=None):
        """Get calib ranges of activation tensors.

        Args:
            q_config (dict, optional): quantization config. Defaults to None.

        Returns:
            dict: calib ranges
        """
        # conduct inference session and get intermediate outputs
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        if sys.version_info < (3, 11) and util.find_spec("onnxruntime_extensions"):
            so.register_custom_ops_library(onnxruntime_extensions.get_library_path())

        execution_provider = (
            self.execution_provider
            if self.execution_provider != "TensorrtExecutionProvider"
            else "CUDAExecutionProvider"
        )
        session = (
            onnxruntime.InferenceSession(self.augmented_model.SerializeToString(), so, providers=[execution_provider])
            if not self.model_wrapper.is_large_model
            else onnxruntime.InferenceSession(
                self.model_wrapper.model_path + "_augment.onnx", so, providers=[execution_provider]
            )
        )

        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
        len_outputs = len(session.get_outputs())
        outputs_names = [session.get_outputs()[i].name for i in range(len_outputs)]

        node_output_names = [
            output.name if output.name not in self.dequantized_output else self.dequantized_output[output.name]
            for output in session.get_outputs()
        ]
        augment_model_wrapper = (
            onnx_model.ONNXModel(self.augmented_model, load_external_data=False)
            if not self.model_wrapper.is_large_model
            else onnx_model.ONNXModel(self.model_wrapper.model_path + "_augment.onnx", load_external_data=False)
        )
        input_name_to_nodes = augment_model_wrapper.input_name_to_nodes()
        output_name_to_node = augment_model_wrapper.output_name_to_node()
        name_to_node = {}
        for data_name in node_output_names:
            node = None
            if data_name in output_name_to_node:
                node = output_name_to_node[data_name]
            elif data_name in input_name_to_nodes:
                node = input_name_to_nodes[data_name][0]
            assert node, "{} is neither an input nor an output of nodes in augmented model.".format(data_name)
            name_to_node[data_name] = node.name

        activation_tensors_calib_range = {}
        intermediate_tensor = {}
        name_to_calibrator = {}
        ort_inputs_for_next_split_model = []

        def _collect_data(inputs):
            for output_idx, output in enumerate(session.run(None, inputs)):
                if q_config is not None and output.size != 0:
                    node_name = name_to_node[node_output_names[output_idx]]
                    if node_output_names[output_idx] not in name_to_calibrator:
                        calib_method = (
                            q_config[node_name]["calibrate_method"] if q_config and node_name in q_config else "MinMax"
                        )
                        assert calib_method in calibrator.CALIBRATOR, "Calibration method {} is not registered.".format(
                            calib_method
                        )
                        _calibrator = calibrator.CALIBRATOR[calib_method]()
                    else:
                        _calibrator = name_to_calibrator[node_output_names[output_idx]]

                    # currently, the calibration range for each iteration is collected if
                    # the calibration method is minmax, otherwise the tensor data is collected.
                    # TODO: for entropy and percentile method, need to support range collection
                    # per iteration in the future.
                    if _calibrator.method_name == "MinMax":
                        _calibrator.collect(output)
                        activation_tensors_calib_range[node_output_names[output_idx]] = [list(_calibrator.calib_range)]
                        name_to_calibrator[node_output_names[output_idx]] = _calibrator
                    else:
                        intermediate_tensor.setdefault((node_output_names[output_idx], node_name), []).append(output)
                elif q_config is None:
                    activation_tensors_calib_range.setdefault(node_output_names[output_idx], []).append(output)

        idx = 0
        while True:
            inputs = self.dataloader.get_next()
            if not inputs:
                break
            if self.iterations != []:
                if idx > max(self.iterations):
                    break
                if idx in self.iterations:
                    _collect_data(inputs)
            else:
                _collect_data(inputs)
            idx += 1

        # for entropy and percentile method, collect calibration range after all tensors are collected.
        merged_dict = intermediate_tensor
        for (output_name, node_name), datas in merged_dict.items():
            if any([data is None for data in datas]):
                continue
            if any([data.dtype in [bool] for data in datas]):  # output type of some ops is bool, skip
                continue
            calib_method = q_config[node_name]["calibrate_method"] if q_config and node_name in q_config else 0
            _calibrator = calibrator.CALIBRATOR[calib_method]()
            _calibrator.collect(datas)
            activation_tensors_calib_range.setdefault(output_name, []).append(list(_calibrator.calib_range))
            _calibrator.clear()
            del _calibrator

        return activation_tensors_calib_range

    def get_weight_tensors_calib_range(self):
        """Get calib ranges of weight tensors.

        Returns:
            dict: calib ranges
        """
        model_nodes_names = [node.name for node in self.model.graph.node]

        # if augmented_model is not None, it means self.white_nodes is already updated in augment_graph func
        # then skip update here
        if self.already_quantized and self.augmented_model is None:
            # mapping between fp32 node and int8 node
            new_white_nodes = []
            for white_node in self.white_nodes:
                new_white_node = white_node + "_quant"
                assert new_white_node in model_nodes_names, "no quantized {} in the " "graph".format(white_node)
                new_white_nodes.append(new_white_node)
            self.white_nodes = new_white_nodes

        added_outputs = set()
        initializer_tensors_to_dump = []
        initializers = [init.name for init in self.model.graph.initializer]
        for node in self.model.graph.node:  # pylint: disable=no-member
            should_be_dump = ((node.op_type in self.dump_op_types) and (node.name not in self.black_nodes)) or (
                node.name in self.white_nodes
            )
            if should_be_dump:
                for input in node.input:
                    if (
                        (self.already_quantized and input.replace("_dequantized", "_quantized") in initializers)
                        or (not self.already_quantized and input in initializers)
                    ) and len(input) != 0:
                        added_outputs.add(input)

        for tensor in added_outputs:
            if tensor not in initializers:
                continue
            if self.augment_nodes:
                for augment_node_type in self.augment_nodes:
                    if augment_node_type in ["DequantizeLinear"]:
                        if not (tensor.endswith("_scale") or tensor.endswith("_zero_point")):
                            initializer_tensors_to_dump.append(tensor)
            else:
                initializer_tensors_to_dump.append(tensor)

        weight_tensors_calib_range = {}
        for initializer_tensor_name in initializer_tensors_to_dump:
            initializer_tensor = self.model_wrapper.get_initializer(initializer_tensor_name)

            # double check initializer tensor is not None
            if initializer_tensor is None:  # pragma: no cover
                continue

            initializer_tensor = onnx.numpy_helper.to_array(
                initializer_tensor,
                base_dir=(
                    os.path.dirname(self.model_wrapper.model_path) if self.model_wrapper.model_path is not None else ""
                ),
            )
            _calibrator = calibrator.CALIBRATOR["MinMax"]()  # use minmax method to calibrate initializer tensors
            if initializer_tensor.flatten().size > 0:
                _calibrator.collect(initializer_tensor)
                weight_tensors_calib_range[initializer_tensor_name] = [list(_calibrator.calib_range)]
            _calibrator.clear()
            del _calibrator
        return weight_tensors_calib_range

    def get_intermediate_outputs(self, q_config=None, activation_only=False, weight_only=False):
        """Gather intermediate model outputs after running inference."""
        output_dicts = {}
        if not activation_only and not weight_only:
            output_dicts = self.get_activation_tensors_calib_range(q_config)
            output_dicts.update(self.get_weight_tensors_calib_range())
        elif weight_only:
            output_dicts = self.get_weight_tensors_calib_range()
        elif activation_only:
            output_dicts = self.get_activation_tensors_calib_range(q_config)

        return list(output_dicts.keys()), output_dicts

    def _dequantize(self, tensor, scale_tensor, zo_tensor):
        """Helper function to dequantize tensor."""
        int_tensor = self.model_wrapper.get_initializer(tensor)
        if int_tensor:  # weight tensor
            return self._dequantize_weight(tensor, scale_tensor, zo_tensor)
        else:
            return self._dequantize_activation(tensor, scale_tensor, zo_tensor)

    def _dequantize_activation(self, activation_tensor_name, scale_tensor, zo_tensor):
        """Helper function to dequantize activation."""
        added_nodes, added_output = self._add_dequantize_node(activation_tensor_name, scale_tensor, zo_tensor)
        self.dequantized_output[added_output] = activation_tensor_name
        return added_nodes, added_output

    def _dequantize_weight(self, weight_tensor_name, scale_tensor, zo_tensor):
        """Helper function to dequantize weight."""
        weight_tensor = self.model_wrapper.get_initializer(weight_tensor_name)
        if len(scale_tensor.dims) in [1, 2] and weight_tensor.dims[0] == max(scale_tensor.dims):
            logger.debug("weight {} is quantized with per channel granularity.".format(weight_tensor_name))
            if self.opset_version < 13 and self.ort_version >= ORT112_VERSION:
                logger.warning(
                    "Skip dequantizing weight {}, please use onnxruntime < 1.12.0 "
                    "or upgrade model opset version to 13 or higher".format(weight_tensor_name)
                )
                return [], None
            node = self.model_wrapper.input_name_to_nodes()[weight_tensor_name][0]
            if "Conv" in node.op_type or ("Gemm" in node.op_type and quant_utils.is_B_transposed(node)):
                added_nodes, added_output = self._add_dequantize_transpose_node(
                    weight_tensor_name, scale_tensor, zo_tensor, len(weight_tensor.dims)
                )
            else:
                added_nodes, added_output = self._add_dequantize_node(
                    weight_tensor_name, scale_tensor, zo_tensor, axis=1 if self.opset_version > 12 else None
                )
        else:
            added_nodes, added_output = self._add_dequantize_node(weight_tensor_name, scale_tensor, zo_tensor)
        self.dequantized_output[added_output] = weight_tensor_name
        return added_nodes, added_output

    def _add_dequantize_node(self, tensor_name, scale_tensor, zo_tensor, axis=None):
        """Helper function to generate dequantize node."""
        dequantize_node = onnx.helper.make_node(
            "DequantizeLinear",
            [tensor_name, scale_tensor.name, zo_tensor.name],
            [tensor_name + "_output"],
            tensor_name + "_DequantizeLinear",
            axis,
        )
        return [dequantize_node], tensor_name + "_output"

    def _add_dequantize_transpose_node(self, tensor_name, scale_tensor, zo_tensor, dim):
        """Insert Transpose-DequantizelLinear-Transpose pairs."""
        pre_transpose_node = onnx.helper.make_node(
            "Transpose",
            inputs=[tensor_name],
            outputs=[tensor_name + "_transposed"],
            perm=(1, 0, 2, 3) if dim == 4 else (1, 0),
            name=tensor_name + "_pre_transpose",
        )
        dequantize_node = onnx.helper.make_node(
            "DequantizeLinear",
            [tensor_name + "_transposed", scale_tensor.name, zo_tensor.name],
            [tensor_name + "_DequantizeLinear"],
            tensor_name + "_DequantizeLinear",
            axis=1 if self.opset_version > 12 else None,
        )
        post_transpose_node = onnx.helper.make_node(
            "Transpose",
            inputs=[tensor_name + "_DequantizeLinear"],
            outputs=[tensor_name + "_output"],
            perm=(1, 0, 2, 3) if dim == 4 else (1, 0),
            name=tensor_name + "_post_transpose",
        )
        added_nodes = [pre_transpose_node, dequantize_node, post_transpose_node]
        return added_nodes, tensor_name + "_output"

    def _map_calibration(self, node_output_names, output_dicts):
        """Map tensor names and min/max values."""
        merged_dict = {}
        for name, minmaxs in output_dicts.items():
            for minmax in minmaxs:
                if len(minmax) < 2:
                    continue
                merged_dict.setdefault(name + "_Min", []).append(minmax[0])
                merged_dict.setdefault(name + "_Max", []).append(minmax[1])

        # Characterizing distribution of a node's values across test data sets
        clean_merged_dict = dict((i, merged_dict[i]) for i in merged_dict)
        pairs = [
            tuple([float(min(clean_merged_dict[name + "_Min"])), float(max(clean_merged_dict[name + "_Max"]))])
            for name in node_output_names
        ]

        final_dict = dict(zip(node_output_names, pairs))
        return final_dict

    def dump_minmax(self, q_config):
        """Get calib ranges of tensors."""
        # pipeline of getting calib ranges of tensors during calibration:
        # 1. augment_graph(): insert activation tensors to model output
        # 2. get_intermediate_outputs():
        #   2.1 get_activation_tensors_calib_range(): get calib ranges of activation tensors using the augment graph
        #   2.2 get_weight_tensors_calib_range(): get calib ranges of weight tensors
        self.augment_graph()
        node_output_names, output_dicts = self.get_intermediate_outputs(q_config)
        return self._map_calibration(node_output_names, output_dicts)

    def dump_calibration(self, q_config, min_max=None):
        """Gather calibration params for quantization.

        Args:
            q_config (dict): op-wise quantization config
            min_max (dict, optional): min/max values of tensors
        """
        return (
            self.calculate_quantization_params(q_config, self.dump_minmax(q_config))
            if min_max is None
            else self.calculate_quantization_params(q_config, min_max)
        )

    def calculate_quantization_params(self, q_config, quantization_thresholds):
        """Given quantization thresholds, calculate the quantization params.

        Args:
            q_config (dict): op-wise quantization config
            quantization_thresholds (dict): Dictionary specifying the min and max values
                                              or outputs of conv and matmul nodes, should be
                                              specified in the following format:
                                              {"param_name": [min, max]}
        """
        if quantization_thresholds is None:
            raise ValueError(
                "quantization thresholds is required to calculate quantization \
                    params (zero point and scale)"
            )

        quantization_params = {}
        model = self.model

        input_name_to_nodes = self.model_wrapper.input_name_to_nodes()
        output_name_to_node = self.model_wrapper.output_name_to_node()

        for tensor_name in quantization_thresholds.keys():
            child = None
            if tensor_name in input_name_to_nodes:
                children = input_name_to_nodes[tensor_name]
                if len(children) == 1:
                    child = children[0]
            parent = None
            sym = False
            qType = 2  # uint8

            # input and output tensor follow activation_type and activation_sym
            if tensor_name in input_name_to_nodes and any(
                [i.name in q_config for i in input_name_to_nodes[tensor_name]]
            ):
                for child in input_name_to_nodes[tensor_name]:
                    if child.name in q_config and q_config[child.name] not in ["fp32", "fp16", "bf16"]:
                        sym = q_config[child.name]["activation_sym"]
                        qType = q_config[child.name]["activation_type"]
                        break
            elif (
                tensor_name in output_name_to_node
                and output_name_to_node[tensor_name].name in q_config
                and q_config[output_name_to_node[tensor_name].name] not in ["fp32", "fp16", "bf16"]
            ):
                sym = q_config[output_name_to_node[tensor_name].name]["activation_sym"]
                qType = q_config[output_name_to_node[tensor_name].name]["activation_type"]
            if self.execution_provider in ["TensorrtExecutionProvider"]:
                # TensorrtExecutionProvider only support int8
                qType = 3
            node_thresholds = quantization_thresholds[tensor_name]
            node_params = self.calculate_scale_zeropoint(
                parent,
                child,
                node_thresholds[0],
                node_thresholds[1],
                sym,
                qType,
            )
            quantization_params[tensor_name] = node_params

        return quantization_params

    def calculate_scale_zeropoint(self, last_node, next_node, rmin, rmax, sym, qType):
        """Given the source and destination node of tensor, return calculated zero point and scales."""
        zp_and_scale = []
        # adjust rmin and rmax such that 0 is included in the range. This is required
        # to make sure zero can be uniquely represented.
        rmin = min(rmin, 0)
        rmax = max(rmax, 0)
        if next_node:
            if next_node.op_type == "Relu":
                if rmin < 0:
                    rmin = 0
            elif next_node.op_type == "Clip" and len(next_node.input) == 3:
                if self.model_wrapper.get_initializer(next_node.input[1]) is not None:
                    clip_min = onnx.numpy_helper.to_array(self.model_wrapper.get_initializer(next_node.input[1]))
                    if rmin < clip_min:
                        rmin = clip_min.tolist() if not isinstance(clip_min.tolist(), list) else clip_min.tolist()[0]
                if self.model_wrapper.get_initializer(next_node.input[2]) is not None:
                    clip_max = onnx.numpy_helper.to_array(self.model_wrapper.get_initializer(next_node.input[2]))
                    if rmax > clip_max:
                        rmax = clip_max.tolist() if not isinstance(clip_max.tolist(), list) else clip_max.tolist()[0]

        if last_node:
            if last_node.op_type in ["Conv", "FusedConv"]:
                attrs = [attr for attr in last_node.attribute]
                attrs_names = [attr.name for attr in last_node.attribute]
                if "activation" in attrs_names:
                    if attrs[attrs_names.index("activation")].s == b"Relu":
                        rmin = max(rmin, 0)
                    if attrs[attrs_names.index("activation")].s == b"Clip":
                        assert (
                            "activation_params" in attrs_names
                        ), "the model contains no params for clip node {}".format(last_node)
                        clip_params = attrs[attrs_names.index("activation_params")].floats
                        rmin = min(rmin, clip_params[0], clip_params[1])
                        rmax = max(rmax, clip_params[0], clip_params[1])

        scale, zp = quant_utils.calculate_scale_zp(rmin, rmax, qType, sym, self.reduce_range)
        zp_and_scale.append(zp)
        zp_and_scale.append(scale)

        return zp_and_scale
