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

from onnx_neural_compressor import constants, data_reader, onnx_model, utility
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.layer_wise import core
from onnx_neural_compressor.algorithms.weight_only import rtn
from onnx_neural_compressor.quantization import config

from typing import List, Union  # isort: skip


def _gptq(
    W: np.array,
    H: np.array,
    num_bits: int = 4,
    group_size: int = 32,
    sym: bool = False,
    block_size: int = 128,
    percdamp: float = 0.01,
    actorder: bool = False,
    mse: bool = False,
    perchannel: bool = True,
):
    """Quant the weight with GPTQ method.

    Args:
        W (np.array): weight.
        H (np.array): Hessian matrix.
        num_bits (int, optional): num_bits. Default is 4.
        group_size (int, optional): how many elements share one scale/zp. Default is 32.
        sym (bool, optional): sym or asym. Defaults to False.
        block_size (int, optional): block_size to quantize weight.
        percdamp (float, optional): percent of the average Hessian diagonal to use for dampening.
        actorder (bool, optional): whether rearrange Hessian matrix considering the diag's value.
        mse (bool, optional): whether get scale and zero point with mse error.
        perchannel (bool, optional): whether quantize weight per-channel.

    Returns:
        Q: fake quantized weight
    """
    Qs = []
    maxq = 2**num_bits - 1
    grid = 100
    maxshrink = 0.8
    norm = 2.4

    def find_params(weight):
        org_shape = weight.shape
        # find zp, scale
        if not perchannel:
            weight = np.expand_dims(weight.flatten(), axis=1)
        tmp = np.zeros(weight.shape[1])
        xmin = np.minimum(np.min(weight, axis=0), tmp)
        xmax = np.maximum(np.max(weight, axis=0), tmp)
        if sym:
            xmax = np.maximum(np.abs(xmin), xmax)
            tmp = xmin < 0
            if np.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / maxq
        if sym:
            zero = np.ones(scale.shape) * (maxq + 1) / 2
        else:
            zero = np.round(-xmin / scale)
        if mse:
            best = np.ones([weight.shape[1]]) * float("inf")
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / maxq
                zero1 = np.round(-xmin1 / scale1) if not sym else zero
                q = np.clip(np.round(weight / scale1) + zero1, 0, maxq)
                q -= weight
                q = np.power(np.abs(q), norm)
                err = np.sum(q, 0)
                tmp = err < best
                if np.any(tmp):
                    best[tmp] = err[tmp]
                    scale[tmp] = scale1[tmp]
                    zero[tmp] = zero1[tmp]
        if not perchannel:
            tmp = org_shape[1]
            scale = np.repeat(scale, tmp)
            zero = np.repeat(zero, tmp)
        shape = [-1] + [1] * (len(org_shape) - 1)
        scale = np.reshape(scale, shape)
        zero = np.reshape(zero, shape)
        return scale, zero

    scales = []
    zps = []
    shape = W.shape
    scale, zp = find_params(W)
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[dead, :] = 0  # such channel makes no contribution to quantization computation

    # rearrange considering the diag's value
    if actorder:
        perm = np.argsort(np.diag(H))[::-1]
        W = W[perm, :]
        H = H[perm, :][:, perm]
    Losses = np.zeros_like(W)
    Q = np.zeros_like(W)
    damp = percdamp * np.mean(np.diag(H))
    diag = np.arange(shape[0])
    H[diag, diag] += damp  # add a average value of
    H = np.linalg.cholesky(np.linalg.inv(H)).T
    Hinv = H
    for i1 in range(0, shape[0], block_size):
        i2 = min(i1 + block_size, shape[0])
        count = i2 - i1

        W1 = copy.deepcopy(W[i1:i2, :])
        Q1 = np.zeros_like(W1)
        Err1 = np.zeros_like(W1)
        Losses1 = np.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):  # within a block, channel wise
            w = W1[i, :]
            d = Hinv1[i, i]

            if group_size != -1:
                if (i1 + i) % group_size == 0:
                    scale, zp = find_params(W[(i1 + i) : (i1 + i + group_size), :])

            q = (scale * (np.clip(np.round(w[:, np.newaxis] / scale) + zp, 0, maxq) - zp)).flatten()
            Q1[i, :] = q
            Losses1[i, :] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[i:, :] -= np.matmul(np.expand_dims(Hinv1[i:, i], axis=1), np.expand_dims(err1, axis=0))
            Err1[i, :] = err1

        Q[i1:i2, :] = Q1
        Losses[i1:i2, :] = Losses1 / 2

        W[i2:, :] -= np.matmul(Hinv[i2:, i1:i2], Err1)

    if actorder:
        invperm = np.argsort(perm)
        Q = Q[invperm, :]

    Q = np.reshape(Q, W.shape)
    del W
    return Q


def gptq_quantize(
    model: Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str],
    data_reader: data_reader.CalibrationDataReader,
    weight_config: dict = {},
    percdamp: float = 0.01,
    block_size: int = 128,
    actorder: bool = False,
    mse: bool = False,
    perchannel: bool = True,
    providers: List[str] = ["CPUExecutionProvider"],
    return_modelproto: bool = True,
):
    """Quant the model with GPTQ method.

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
                    }. Defaults to {}.
        percdamp (float, optional): percentage of Hessian's diagonal values' average, which will be added
            to Hessian's diagonal to increase numerical stability. Defaults to 0.01.
        block_size (int, optional): execute GPTQ quantization per block. Defaults to 128.
        actorder (bool, optional): whether to sort Hessian's diagonal values to rearrange channel-wise
            quantization order. Defaults to False.
        mse (bool, optional): whether get scale and zero point with mse error. Defaults to False.
        perchannel (bool, optional): whether quantize weight per-channel. Defaults to True.
        providers (list, optional): providers to use. Defaults to ["CPUExecutionProvider"].
        return_modelproto (bool, optionmal): whether to return onnx.Modelproto. set False for layer-wise quant.
            Default to True

    Returns:
        onnx.ModelProto: quantized onnx model
    """
    if not isinstance(model, onnx_model.ONNXModel):
        model = onnx_model.ONNXModel(model)
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""

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
    if model.is_large_model:
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

    input_name_to_nodes = model.input_name_to_nodes()

    for idx, input_name in enumerate(output_names):
        utility.simple_progress_bar(len(output_names), idx + 1)
        node_list = []
        weights = []

        for node in input_name_to_nodes[input_name]:
            # check op_type of node is MatMul
            # check op_name in quantization config
            # check dim 1 of input is weight tensor
            if (
                node.op_type in ["MatMul"]
                and node.name in weight_config
                and model.get_initializer(node.input[1]) is not None
            ):
                weight = onnx.numpy_helper.to_array(
                    model.get_initializer(model.get_node(node.name).input[1]), base_dir
                ).copy()
                if len(weight.shape) != 2:  # pragma: no cover
                    continue

                weights.append(weight)
                node_list.append(model.get_node(node.name))

        if len(weights) == 0:  # pragma: no cover
            continue

        Hs = [np.zeros((i.shape[0], i.shape[0])) for i in weights]
        nsamples = 0
        for data in inputs:
            inp = session.run([input_name], data)[0]
            tmp = inp.shape[0]
            inp = np.reshape(inp, (-1, inp.shape[-1]))
            Hs = [i * (nsamples / (nsamples + tmp)) for i in Hs]
            nsamples += tmp
            inp = np.sqrt(2 / nsamples) * inp
            Hs = [i + np.matmul(inp.T, inp) for i in Hs]

        for (
            node,
            weight,
            H,
        ) in zip(node_list, weights, Hs):
            num_bits = weight_config[node.name].get("weight_bits", 4)
            group_size = weight_config[node.name].get("weight_group_size", 32)
            sym = weight_config[node.name].get("weight_sym", True)
            dtype = weight_config[node.name].get("weight_dtype", "int")
            accuracy_level = weight_config[node.name].get("accuracy_level", 0)
            quant_format = getattr(weight_config[node.name].get("quant_format", None), "value", None)

            weight_tensor = model.get_initializer(node.input[1])
            init_share_num = model.get_initializer_share_num(node.input[1])

            # weight -> quant -> dequant -> q_weight
            q_weight = _gptq(
                weight,
                H,
                num_bits=num_bits,
                group_size=group_size,
                sym=sym,
                block_size=block_size,
                percdamp=percdamp,
                actorder=actorder,
                mse=mse,
                perchannel=perchannel,
            )
            new_nodes, new_inits, remove_nodes = quant_utils.quant_matmul_weight_only(
                node=node,
                weight=weight,
                dtype=dtype,
                num_bits=num_bits,
                sym=sym,
                group_size=group_size,
                quant_format=quant_format,
                accuracy_level=accuracy_level,
            )
            model.add_initializers(new_inits)
            model.add_nodes(new_nodes)
            model.remove_nodes(remove_nodes)

            if init_share_num == 1:
                model.remove_initializer(weight_tensor)

    model.remove_tensors_from_outputs(output_names)
    model.model.graph.output.MergeFrom(org_output)
    model.topological_sort()

    # reload external data to prevent external data file path errors
    if model.is_large_model:

        onnx.external_data_helper.load_external_data_for_model(model.model, os.path.split(model.model_path)[0])

    if return_modelproto:
        return model.model
    else:
        model.save(model.model_path + "_quant.onnx")
        return model


def apply_gptq_on_model(
    model: Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str],
    quant_config: dict,
    calibration_data_reader: data_reader.CalibrationDataReader,
    percdamp: float = 0.01,
    block_size: int = 128,
    actorder: bool = False,
    mse: bool = False,
    perchannel: bool = True,
    providers: List[str] = ["CPUExecutionProvider"],
    layer_wise_quant: bool = False,
) -> onnx.ModelProto:
    """Apply GPTQ on onnx model.

    Args:
        model (Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str]): onnx model.
        quant_config (dict): quantization config.
        calibration_data_reader (data_reader.CalibrationDataReader): data_reader for calibration.

    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    # set other model params
    quant_kwargs = {
        "percdamp": percdamp,
        "block_size": block_size,
        "actorder": actorder,
        "mse": mse,
        "perchannel": perchannel,
        "providers": providers,
    }

    if layer_wise_quant:
        quantized_model = core.layer_wise_quant(
            model,
            quant_func=gptq_quantize,
            weight_config=quant_config,
            data_reader=calibration_data_reader,
            **quant_kwargs
        )
    else:
        quantized_model = gptq_quantize(
            model, data_reader=calibration_data_reader, weight_config=quant_config, **quant_kwargs
        )

    if isinstance(quantized_model, onnx_model.ONNXModel):
        quantized_model = quantized_model.model
    quant_utils.dump_woq_stats(quantized_model, quant_config)
    return quantized_model
