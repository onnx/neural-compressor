# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2024 Intel Corporation
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

import os
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
from packaging import version

from onnx_neural_compressor import constants, onnx_model, utility
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.layer_wise import core

from typing import List, Union  # isort: skip


def rtn_quantize(
    model: Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str],
    weight_config: dict = {},
    ratios: dict = {},
    providers: List[str] = ["CPUExecutionProvider"],
    return_modelproto: bool = True,
):
    """Quantize the model with round to nearst method.

    Args:
        model (Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str]): onnx model
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
        ratios (dict, optional): percentile of clip. Defaults to {}.
        providers (list, optional): providers to use. Defaults to ["CPUExecutionProvider"].
        return_modelproto (bool, optionmal): whether to return onnx.Modelproto. set False for layer-wise quant.
            Default to True
    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    if not isinstance(model, onnx_model.ONNXModel):
        model = onnx_model.ONNXModel(model)
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""
    new_nodes = []
    remove_nodes = []
    total_num = len([i for i in model.nodes() if i.op_type in ["MatMul"]])
    curr_id = 0
    for node in model.nodes():
        if node.op_type in ["MatMul"]:
            curr_id += 1
            utility.simple_progress_bar(total_num, curr_id)

        # check op_type of node is MatMul
        # check dim 1 of input is weight tensor
        # check weight_type is not "fp32"
        if (
            node.op_type in ["MatMul"]  # check op_type of node is MatMul
            and model.get_initializer(node.input[1]) is not None
            and weight_config.get(node.name, {}).get("weight_dtype", "fp32") != "fp32"
        ):
            weight_tensor = model.get_initializer(node.input[1])
            weight = onnx.numpy_helper.to_array(weight_tensor, base_dir=base_dir).copy()
            if len(weight.shape) != 2:
                continue

            dtype = weight.dtype
            num_bits = weight_config[node.name].get("weight_bits", 4)
            group_size = weight_config[node.name].get("weight_group_size", 32)
            sym = weight_config[node.name].get("weight_sym", True)
            accuracy_level = weight_config[node.name].get("accuracy_level", 0)

            org_w_shape = weight.shape  # ic, oc
            group_size = group_size if group_size != -1 else org_w_shape[0]

            k_blocks = (org_w_shape[0] - 1) // group_size + 1
            init_share_num = model.get_initializer_share_num(node.input[1])

            weight = quant_utils.pad_tensor(weight, group_size, k_blocks)

            satisfy_MatMulNBits_condition = (
                version.Version(ort.__version__) > constants.ONNXRT1161_VERSION and num_bits == 4
            )
            satisfy_MatMulFpQ4_condition = (
                version.Version(ort.__version__) >= constants.ONNXRT116_VERSION and num_bits == 4 and group_size == 32
            )
            if model.model.opset_import[0].version <= 20:
                _, _, zp, scale, q_weight =quant_utils.quantize_data(
                    weight.T.reshape((-1, group_size)),
                    "uint" + str(num_bits),
                    sym,
                    ratio=ratios.get(node.input[1], 1),
                    axis=1,
                )
                dequant_node, new_inits =quant_utils.make_weight_only_dequant_node(
                    node=node,
                    num_bits=num_bits,
                    k_blocks=k_blocks,
                    q_weight=q_weight.reshape(weight.T.shape).T,
                    scale=scale.astype(dtype),
                    axis=1,
                    block_size=group_size,
                    zero_point=zp,
                )
                model.add_initializers(new_inits)
                new_nodes.append(dequant_node)
            elif ("CUDAExecutionProvider" in providers and satisfy_MatMulNBits_condition) or (
                "CUDAExecutionProvider" not in providers
                and (satisfy_MatMulFpQ4_condition or satisfy_MatMulNBits_condition)
            ):  # pragma: no cover
                # MatMulFpQ4 support 4 bits and 32 group_size with ort 1.16.0 and 1.16.1 versions, supported by CPU EP
                # MatMulNBits supports 4 bits and 2^n group_size with ort > 1.16.1, supported by CPU EP AND CUDA EP
                _, _, zp, scale, q_weight = quant_utils.quantize_data(
                    weight.T.reshape((-1, group_size)),
                    "uint" + str(num_bits),
                    sym,
                    ratio=ratios.get(node.input[1], 1),
                    axis=1,
                )
                q_matmul_node, new_inits = quant_utils.make_matmul_weight_only_node(
                    node=node,
                    weight_shape=org_w_shape,
                    num_bits=num_bits,
                    group_size=group_size,
                    k_blocks=k_blocks,
                    q_weight=q_weight,
                    scale=scale.astype(dtype),
                    zero_point=zp if not sym else None,
                    accuracy_level=accuracy_level,
                )

                model.add_initializers(new_inits)
                remove_nodes.append(node)
                new_nodes.append(q_matmul_node)
            else:
                q_weight = quant_utils.qdq_data(
                    weight.T.reshape((-1, group_size)),
                    "int" + str(num_bits),
                    sym,
                    ratio=ratios.get(node.input[1], 1),
                    axis=1,
                )
                q_weight = np.reshape(q_weight, (org_w_shape[1], -1))
                q_weight = np.transpose(q_weight)
                q_weight = q_weight[: org_w_shape[0], :].astype(dtype)
                q_weight_tensor = onnx.helper.make_tensor(
                    name=node.input[1] + "_Q{}G{}".format(str(num_bits), str(group_size)),
                    data_type=onnx.helper.np_dtype_to_tensor_dtype(dtype),
                    dims=weight.shape,
                    vals=q_weight.tobytes(),
                    raw=True,
                )
                model.add_initializer(q_weight_tensor)
                node.input[1] = q_weight_tensor.name
            if init_share_num == 1:
                model.remove_initializer(weight_tensor)

    model.add_nodes(new_nodes)
    model.remove_nodes(remove_nodes)
    model.topological_sort()

    # reload external data to prevent external data file path errors
    if model.is_large_model:
        onnx.external_data_helper.load_external_data_for_model(model.model, os.path.split(model.model_path)[0])

    if return_modelproto:
        return model.model
    else:
        model.save(model.model_path + "_quant.onnx")
        return model


def apply_rtn_on_model(
    model: Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str],
    quant_config: dict,
    ratios: dict = {},
    providers: List[str] = ["CPUExecutionProvider"],
    layer_wise_quant: bool = False,
) -> onnx.ModelProto:
    """Apply RTN on onnx model.

    Args:
        model (Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str]): onnx model.
        quant_config (dict): quantization config.

    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    quant_kwargs = {
        "ratios": ratios,
        "providers": providers,
    }

    if layer_wise_quant:
        quantized_model = core.layer_wise_quant(
            model, quant_func=rtn_quantize, weight_config=quant_config, **quant_kwargs
        )
    else:
        quantized_model = rtn_quantize(model, weight_config=quant_config, **quant_kwargs)

    if isinstance(quantized_model, onnx_model.ONNXModel):
        quantized_model = quantized_model.model
    quant_utils.dump_woq_stats(quantized_model, quant_config)
    return quantized_model
