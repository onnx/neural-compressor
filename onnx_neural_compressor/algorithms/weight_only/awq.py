# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
from packaging import version

from onnx_neural_compressor import constants, data_reader, logger, onnx_model
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.weight_only import rtn

from typing import List, Union  # isort: skip


def _get_weight_scale(weight, group_size):
    """Get the scale of weight."""
    org_shape = weight.shape
    weight = np.reshape(weight, (-1, group_size)) if group_size != -1 else weight
    scale = np.mean(np.reshape(np.abs(weight) / np.max(np.abs(weight), axis=1, keepdims=True), org_shape), axis=0)
    return scale


def _apply_awq_scale(model, weight_config, absorb_pairs, output_dicts):
    """Apply scale for salient weight."""
    best_scales = {}
    new_init_tensors = []
    new_added_mul_nodes = []
    replace_input = []
    updated_nodes = []
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""

    input_name_to_nodes = model.input_name_to_nodes()
    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]):  # pragma: no cover
            logger.warning(
                "Miss input tensors of nodes {} during AWQ, skip it!".format(
                    ", ".join([node.name for node in nodes if node.input[0] not in output_dicts])
                )
            )
            continue
        inp = np.concatenate(output_dicts[nodes[0].input[0]], axis=0)
        inp_scale = np.mean(np.reshape(np.abs(inp), (-1, inp[0].shape[-1])), axis=0)
        dtype = None
        weight = []
        org_out = []

        weight_dtype = weight_config[nodes[0].name].get("weight_dtype", "int")
        num_bits = weight_config[nodes[0].name].get("weight_bits", 4)
        group_size = weight_config[nodes[0].name].get("weight_group_size", 32)
        sym = weight_config[nodes[0].name].get("weight_sym", True)
        accuracy_level = weight_config[nodes[0].name].get("accuracy_level", 0)

        # use same params for all children of one parent
        for node in nodes:
            weight_config.setdefault(node.name, {}).update({"weight_dtype": weight_dtype})
            weight_config.setdefault(node.name, {}).update({"weight_bits": num_bits})
            weight_config.setdefault(node.name, {}).update({"weight_group_size": group_size})
            weight_config.setdefault(node.name, {}).update({"weight_sym": sym})

        # search scale
        best_error = float("inf")
        best_ratio = -1
        best_scale = None
        n_grid = 20

        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            loss = 0
            for node in nodes:
                weight = onnx.numpy_helper.to_array(model.get_initializer(node.input[1]), base_dir)
                if len(weight.shape) != 2:
                    continue

                org_out = np.matmul(inp, weight)
                org_w_shape = weight.shape
                group_size = group_size if group_size != -1 else org_w_shape[0]

                w_scale = _get_weight_scale(weight.T, weight.shape[0])
                scales = np.clip(np.power(inp_scale, ratio) / np.power(w_scale, (1 - ratio)), 1e-4, None)
                scales = scales / np.sqrt(np.max(scales) * np.min(scales))
                weight = weight.T * scales
                weight = quant_utils.pad_tensor(weight.T, group_size, (org_w_shape[0] + group_size - 1) // group_size)

                q_weight = quant_utils.qdq_data(
                    weight.reshape((-1, group_size)),
                    weight_dtype + str(num_bits),
                    sym,
                ).reshape(weight.shape)

                q_weight = q_weight[: org_w_shape[0], :] / np.expand_dims(scales, axis=-1)
                out = np.matmul(inp, q_weight)
                loss += np.mean(np.power((org_out - out), 2))

            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

        for node in nodes:
            init_share_num = model.get_initializer_share_num(node.input[1])
            weight_tensor = model.get_initializer(node.input[1])
            tensor = onnx.numpy_helper.to_array(weight_tensor, base_dir)
            dtype = tensor.dtype
            tensor = tensor.T * best_scale
            tensor = (tensor.T).astype(dtype)

            new_tensor = onnx.helper.make_tensor(
                name=node.input[1] + "_scaled",
                data_type=onnx.helper.np_dtype_to_tensor_dtype(dtype),
                dims=tensor.shape,
                vals=tensor.tobytes(),
                raw=True,
            )
            model.add_initializer(new_tensor)
            node.input[1] = new_tensor.name

            if init_share_num == 1:
                model.remove_initializer(weight_tensor)

        if parent is None:
            continue
        parent = model.get_node(parent)
        if parent is None or parent.name in updated_nodes:
            continue

        if parent.op_type in ["LayerNormalization", "BatchNormalization", "InstanceNormalization"] and len(
            input_name_to_nodes[nodes[0].input[0]]
        ) == len(
            nodes
        ):  # pragma: no cover
            for idx in [1, 2]:
                tensor = onnx.numpy_helper.to_array(model.get_initializer(parent.input[idx]), base_dir)
                dtype = tensor.dtype
                new_tensor = tensor / np.reshape(best_scale, (1, -1))
                model.set_initializer(parent.input[idx], new_tensor.astype(dtype), raw=True)
                updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif (
            parent.op_type in ["SimplifiedLayerNormalization", "MatMul", "Gemm", "Mul"]
            and not all([model.get_initializer(inp) is None for inp in parent.input])
            and len(input_name_to_nodes[nodes[0].input[0]]) == len(nodes)
        ):  # pragma: no cover
            for inp in parent.input:
                if model.get_initializer(inp) is not None:
                    tensor = onnx.numpy_helper.to_array(model.get_initializer(inp), base_dir)
                    dtype = tensor.dtype
                    new_tensor = tensor / np.reshape(best_scale, (1, -1))
                    model.set_initializer(inp, new_tensor.astype(dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        elif parent.op_type in ["Conv", "FusedConv"] and len(input_name_to_nodes[nodes[0].input[0]]) == len(
            nodes
        ):  # pragma: no cover
            tensor = onnx.numpy_helper.to_array(model.get_initializer(parent.input[2]), base_dir)
            dtype = tensor.dtype
            new_tensor = tensor / np.reshape(best_scale, (1, -1))
            model.set_initializer(parent.input[2], new_tensor.astype(dtype), raw=True)
            updated_nodes.append(parent.name)
            output_dicts[parent.output[0]] = output_dicts[parent.output[0]] / np.reshape(best_scale, (1, -1))

        else:
            # insert mul
            scale_tensor = onnx.helper.make_tensor(
                name=parent.output[0] + "_weight_only_scale",
                data_type=onnx.helper.np_dtype_to_tensor_dtype(dtype),
                dims=best_scale.shape,
                vals=(1.0 / best_scale).flatten().tolist(),
            )
            new_init_tensors.append(scale_tensor)
            mul_output_name = parent.output[0] + "_weight_only_out"
            mul_node = onnx.helper.make_node(
                "Mul",
                inputs=[nodes[0].input[0], scale_tensor.name],
                outputs=[mul_output_name],
                name=nodes[0].input[0] + "_weight_only_mul",
            )
            new_added_mul_nodes.append(mul_node)
            for node in nodes:
                replace_input.append([node, node.input[0], mul_node.output[0]])
            updated_nodes.append(parent.name)
            output_dicts[mul_node.output[0]] = output_dicts[mul_node.input[0]] / np.reshape(best_scale, (1, -1))

    model.add_nodes(new_added_mul_nodes)
    model.add_initializers(new_init_tensors)
    for node, old_input_name, new_input_name in replace_input:
        model.replace_node_input(node, old_input_name, new_input_name)

    return model, output_dicts


def _apply_awq_clip(model, weight_config, absorb_pairs, output_dicts):
    """Apply clip for weight by checking mse."""
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""
    ratios = {}
    for parent, nodes in absorb_pairs.items():
        if any([node.input[0] not in output_dicts for node in nodes]):
            logger.warning(
                "Miss input tensors of nodes {} during AWQ, skip it!".format(
                    ", ".join([node.name for node in nodes if node.input[0] not in output_dicts])
                )
            )
            continue

        inp = np.concatenate(output_dicts[nodes[0].input[0]], axis=0)

        for node in nodes:
            weight_dtype = weight_config[node.name].get("weight_dtype", "int")
            num_bits = weight_config[node.name].get("weight_bits", 4)
            group_size = weight_config[node.name].get("weight_group_size", 32)
            sym = weight_config[node.name].get("weight_sym", True)
            accuracy_level = weight_config[node.name].get("accuracy_level", 0)

            org_weight = onnx.numpy_helper.to_array(model.get_initializer(node.input[1]), base_dir=base_dir)
            org_w_shape = org_weight.shape  # ic, oc
            group_size = group_size if group_size != -1 else org_w_shape[0]
            org_out = np.matmul(inp, org_weight)  # n_token, oc
            k_blocks = (org_w_shape[0] - 1) // group_size + 1
            org_weight = quant_utils.pad_tensor(org_weight, group_size, k_blocks)

            org_weight = np.transpose(org_weight)

            best_error = float("inf")
            best_ratio = 1
            for i_s in range(10):
                ratio = 1 - i_s / 100
                weight = copy.deepcopy(org_weight)
                weight = quant_utils.qdq_data(
                    weight.reshape((-1, group_size)),
                    weight_dtype + str(num_bits),
                    sym,
                    ratio=ratio,
                ).reshape(org_weight.shape)

                cur_out = np.matmul(inp, weight[:, : org_w_shape[0]].T)
                loss = np.mean(np.power((org_out - cur_out), 2))
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
            ratios[node.input[1]] = best_ratio
    return ratios


def awq_quantize(
    model: Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str],
    data_reader: data_reader.CalibrationDataReader,
    weight_config: dict = {},
    enable_auto_scale: bool = True,
    enable_mse_search: bool = True,
    providers: List[str] = ["CPUExecutionProvider"],
) -> onnx.ModelProto:
    """Quant the model with Activation-aware Weight quantization(AWQ) method.

    Args:
        model (Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str]): onnx model.
        data_reader (data_reader.CalibrationDataReader): data_reader for calibration.
        weight_config (dict, optional): quantization config
            For example,
            weight_config = {
                '(fc2, "MatMul")':
                {
                    'weight_dtype': 'int',
                    'weight_bits': 4,
                    'weight_group_size': 32,
                    'weight_sym': True,
                    'accuracy_level': 0
                }
            }. Defaults to {}.
        enable_auto_scale (bool, optional): whether to search for best scales based on activation
            distribution. Defaults to True.
        enable_mse_search (bool, optional): whether to search for the best clip range from range
            [0.91, 1.0, 0.01]. Defaults to True.
        providers (list, optional): providers to use. Defaults to ["CPUExecutionProvider"].

    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    if not isinstance(model, onnx_model.ONNXModel):
        model = onnx_model.ONNXModel(model)
    output_dicts = {}
    full_ratio = {}

    if enable_mse_search:
        inputs, so = quant_utils.prepare_inputs(model, data_reader, providers)
        del data_reader

        org_output = copy.deepcopy(model.model.graph.output)
        model.remove_tensors_from_outputs([i.name for i in org_output])

        output_names = []
        for node in model.nodes():
            # check op_type of node is MatMul
            # check op_name in quantization config
            # check dim 1 of input is weight tensor
            if (
                node.op_type in ["MatMul"]
                and node.name in weight_config
                and model.get_initializer(node.input[1]) is not None
            ):
                output_names.append(node.input[0])
        output_names = list(set(output_names))
        model.add_tensors_to_outputs(output_names)

        if model.is_large_model:  # pragma: no cover
            onnx.save_model(
                model.model,
                model.model_path + "_augment.onnx",
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )

        session = (
            ort.InferenceSession(model.model.SerializeToString(), so, providers=providers)
            if not model.is_large_model
            else ort.InferenceSession(model.model_path + "_augment.onnx", so, providers=providers)
        )

        output_name_to_node = model.output_name_to_node()
        input_name_to_nodes = model.input_name_to_nodes()
        for input_name in output_names:
            # input_name maybe the input of graph and there is no parent node
            parent = output_name_to_node[input_name].name if input_name in output_name_to_node else None
            dump_pairs = {parent: []}

            for node in input_name_to_nodes[input_name]:
                # check op_type of node is MatMul
                # check op_name in quantization config
                # check dim 1 of input is weight tensor
                if (
                    node.op_type in ["MatMul"]
                    and node.name in weight_config
                    and model.get_initializer(node.input[1]) is not None
                ):
                    dump_pairs[parent].append(model.get_node(node.name))

            if len(dump_pairs[parent]) == 0:  # pragma: no cover
                continue

            output_dicts = {}
            for inp in inputs:
                output = session.run([input_name], inp)
                output_dicts.setdefault(input_name, []).append(output)

            if enable_auto_scale:
                model, output_dicts = _apply_awq_scale(
                    model,
                    weight_config,
                    dump_pairs,
                    output_dicts,
                )
            if enable_mse_search:
                ratios = _apply_awq_clip(
                    model,
                    weight_config,
                    dump_pairs,
                    output_dicts,
                )
            del output_dicts
            del dump_pairs
            full_ratio.update(ratios)

        model.remove_tensors_from_outputs(output_names)
        model.model.graph.output.MergeFrom(org_output)
    model = rtn.rtn_quantize(
        model=model,
        weight_config=weight_config,
        ratios=full_ratio,
        providers=providers,
    )
    return model


def apply_awq_on_model(
    model: Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str],
    quant_config: dict,
    calibration_data_reader: data_reader.CalibrationDataReader,
    enable_auto_scale: bool = True,
    enable_mse_search: bool = True,
    providers: List[str] = ["CPUExecutionProvider"],
) -> onnx.ModelProto:
    """Apply Activation-aware Weight quantization(AWQ) on onnx model.

    Args:
        model (Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str]): nnx model.
        quant_config (dict): quantization config.
        calibration_data_reader (data_reader.CalibrationDataReader): data_reader for calibration.

    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    # set model params
    kwargs = {
        "enable_auto_scale": enable_auto_scale,
        "enable_mse_search": enable_mse_search,
        "providers": providers,
    }
    q_model = awq_quantize(model, data_reader=calibration_data_reader, weight_config=quant_config, **kwargs)
    quant_utils.dump_woq_stats(q_model, quant_config)
    return q_model
