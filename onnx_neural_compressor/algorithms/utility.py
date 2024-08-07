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

import re
import struct
import sys
from importlib import util

import numpy as np
from packaging import version

from onnx_neural_compressor import constants, utility

if sys.version_info < (3, 11) and util.find_spec("onnxruntime_extensions"):  # pragma: no cover
    import onnxruntime_extensions

onnx = utility.LazyImport("onnx")
ort = utility.LazyImport("onnxruntime")

__producer__ = "onnx.quantize"
__version__ = "0.1.0"
onnx_domain = "ai.onnx"
ms_domain = "com.microsoft"
QUANT_OP_NAME_SUFFIX = "_quant"


def attribute_to_kwarg(attribute):
    """Convert attribute to kwarg format for use with onnx.helper.make_node."""
    attribute_mapping = {
        1: attribute.f,
        2: attribute.i,
        3: attribute.s,
        4: attribute.t,
        5: attribute.g,
        6: attribute.floats,
        7: attribute.ints,
        8: attribute.strings,
        9: attribute.tensors,
        10: attribute.graphs,
    }
    if attribute.type in attribute_mapping:
        value = attribute_mapping[attribute.type]
    else:  # pragma: no cover
        raise ValueError(
            "attribute {} has no type specified " "or unsupported type {}.".format(attribute.name, attribute.type)
        )
    return {attribute.name: value}


ONNX_INT_TYPE_RANGE = {
    onnx.TensorProto.UINT8: (0, 255),
    onnx.TensorProto.INT8: (-128, 127),
}

ONNX_INT_TYPE_SYMMETRIC_RANGE = {
    onnx.TensorProto.INT8: (-127, 127),
}

ONNX_INT_TYPE_REDUCED_RANGE = {
    onnx.TensorProto.UINT8: (0, 127),
    onnx.TensorProto.INT8: (-64, 64),
}

ONNX_STR_TYPE_RANGE = {
    "int1": (-1, 0),
    "int2": (-2, 1),
    "int3": (-4, 3),
    "int4": (-8, 7),  # onnx >= 1.16.0 defines TensorProto.INT4
    "int5": (-16, 15),
    "int6": (-32, 31),
    "int7": (-64, 63),
    "int8": (-128, 127),
    "uint1": (0, 1),
    "uint2": (0, 3),
    "uint3": (0, 7),
    "uint4": (0, 15),  # onnx >= 1.16.0 defines TensorProto.UINT4
    "uint5": (0, 31),
    "uint6": (0, 63),
    "uint7": (0, 127),
    "uint8": (0, 255),
}


def _qType_to_np_type(qType):
    if isinstance(qType, int):
        return onnx.helper.tensor_dtype_to_np_dtype(qType)
    elif isinstance(qType, str) and "uint" in qType:
        return np.dtype("uint8")
    else:
        return np.dtype("int8")


def find_by_name(name, item_list):
    """Helper function to find item by name in a list."""
    items = []
    for item in item_list:
        assert hasattr(item, "name"), "{} should have a 'name' attribute defined".format(item)
        if item.name == name:
            items.append(item)
    if len(items) > 0:
        return items[0]
    else:
        return None


def get_qmin_qmax_for_qType(qType, reduce_range=False, sym=False):  # noqa: N802
    """Get qmin, qmax for qType.

    Args:
        qType (int or str): int for onnx defined type, str for onnx not defined type
        reduce_range (bool, optional): whether use 7 bit for 8bit quantization
        sym (bool, optional): quantization scheme. Defaults to False.
    """
    if qType == onnx.TensorProto.FLOAT8E4M3FN:
        raise NotImplementedError("This function is not implemented for float 8 as not needed.")

    qrange = None

    if isinstance(qType, str):
        qrange = ONNX_STR_TYPE_RANGE.get(qType)
    elif reduce_range:
        qrange = ONNX_INT_TYPE_REDUCED_RANGE.get(qType)
    elif sym and qType in ONNX_INT_TYPE_SYMMETRIC_RANGE:
        qrange = ONNX_INT_TYPE_SYMMETRIC_RANGE[qType]
    else:
        qrange = ONNX_INT_TYPE_RANGE.get(qType)

    if not qrange:
        raise ValueError(f"Unexpected data type {qType} requested.")

    return qrange


def quantize_nparray(dtype, arr, scale, zero_point, low=None, high=None):
    """Quantize numpy array."""
    q_weight = np.empty_like(np.asarray(arr), dtype=np.asarray(scale).dtype)
    np.divide(arr, scale, out=q_weight)
    np.add(q_weight, zero_point, out=q_weight)
    np.round(q_weight, out=q_weight)
    if low is not None and high is not None:
        np.clip(q_weight, low, high, out=q_weight)
    return q_weight.astype(dtype)


def quantize_data_per_channel(data, axis, qType, sym, reduce_range=False):
    """Quantize tensor per-channel."""
    quantize_range = get_qmin_qmax_for_qType(qType, reduce_range, sym)
    rmin = None
    rmax = None
    for i in range(len(data.shape)):
        if i != axis:
            rmin = np.min(data, axis=i, keepdims=True) if rmin is None else np.min(rmin, axis=i, keepdims=True)
            rmax = np.max(data, axis=i, keepdims=True) if rmax is None else np.max(rmax, axis=i, keepdims=True)
    rmin = np.minimum(rmin, 0)
    rmax = np.maximum(rmax, 0)
    scale, zero_point = calculate_scale_zp(rmin, rmax, qType, sym, reduce_range)

    dtype = _qType_to_np_type(qType)
    quantized_data = quantize_nparray(dtype, data, scale, zero_point, low=quantize_range[0], high=quantize_range[1])
    return rmin.reshape(-1, 1), rmax.reshape(-1, 1), zero_point.reshape(-1, 1), scale.reshape(-1, 1), quantized_data


def dequantize_data_with_scale_zero(tensor_value, scale_value, zo_value):
    """Dequantize tensor with scale and zero point."""
    return (tensor_value.astype(scale_value.dtype) - zo_value.astype(scale_value.dtype)) * scale_value


def dequantize_data(tensor_value, scale_value, zo_value, axis=0):
    """Dequantize tensor."""
    if not isinstance(scale_value, np.ndarray):
        return dequantize_data_with_scale_zero(tensor_value, scale_value, zo_value)
    else:
        channel_count = tensor_value.shape[axis]  # TBD, default from axis 0
        new_per_channel_tensor_values = []
        for i in range(channel_count):
            per_channel_tensor_value = tensor_value.take(i, axis)
            per_channel_scale_value = scale_value.take(i)
            per_channel_zero_value = zo_value.take(i)
            new_per_channel_tensor_values.append(
                dequantize_data_with_scale_zero(
                    per_channel_tensor_value, per_channel_scale_value, per_channel_zero_value
                )
            )
        # combine per_channel_data into one
        reshape_dims = list(tensor_value.shape)  # deep copy
        reshape_dims[axis] = 1  # only one per channel for reshape
        new_tensor_value = new_per_channel_tensor_values[0].reshape(reshape_dims)
        for i in range(1, channel_count):
            new_per_channel_tensor_value = new_per_channel_tensor_values[i].reshape(reshape_dims)
            new_tensor_value = np.concatenate((new_tensor_value, new_per_channel_tensor_value), axis)
        return new_tensor_value


def calculate_scale_zp(rmin, rmax, qType, sym, reduce_range=False):
    """Calculate scale and zero point."""
    qmin, qmax = get_qmin_qmax_for_qType(qType, reduce_range, sym)
    dtype = _qType_to_np_type(qType)
    if isinstance(rmax, np.ndarray):
        if sym:
            max_range = np.maximum(abs(rmin), abs(rmax))
            rmin = -max_range
            rmax = max_range
        scale = (rmax - rmin) / (qmax - qmin)
        scale[scale < np.finfo(rmax.dtype).tiny] = 1
        zero_point = (
            np.multiply(np.ones(rmax.shape), np.round((qmax + qmin) / 2.0)).astype(dtype)
            if sym
            else np.round(qmin - rmin / scale).astype(dtype)
        )
    else:
        if sym:
            max_range = max(abs(rmin), abs(rmax))
            scale = (float(max_range) * 2) / (qmax - qmin) if max_range > 0 else 1
        else:
            scale = (float(rmax) - float(rmin)) / (qmax - qmin) if rmin != rmax else 1
        zero_point = np.round((qmax + qmin) / 2.0).astype(dtype) if sym else np.round(qmin - rmin / scale).astype(dtype)
    return np.float32(scale), zero_point


def quantize_data(data, qType, sym, reduce_range=False, ratio=1.0, axis=None):
    """Quantize data.

    To pack weights, we compute a linear transformation
        - when data type == uint8 mode, from [rmin, rmax] -> [0, 2^{b-1}] and
        - when data type == int8, from [-m , m] -> [-(2^{b-1}-1), 2^{b-1}-1] where
            m = max(abs(rmin), abs(rmax))
    and add necessary intermediate nodes to transform quantized weight to full weight
    using the equation r = S(q-z), where
        r: real original value
        q: quantized value
        S: scale
        z: zero point

    Args:
        data (array): data to quantize
        qType (int): data type to quantize to. Supported types UINT8 and INT8
        sym (bool): whether use sym quantization.
        reduce_range (bool): whether use 7 bit or not. Defaults to False
        ratio (float, optional): percentile of clip. Defaults to 1.0
        axis (int, optional): process data along a specific axis. Default is None (process the whole data)
    """
    quantize_range = get_qmin_qmax_for_qType(qType, reduce_range, sym)
    rmin = np.min(np.min(data), 0) if axis is None else np.min(data, axis=axis, keepdims=True)
    rmax = np.max(np.max(data), 0) if axis is None else np.max(data, axis=axis, keepdims=True)
    rmin *= ratio
    rmax *= ratio

    scale, zero_point = calculate_scale_zp(rmin, rmax, qType, sym, reduce_range)
    dtype = _qType_to_np_type(qType)
    quantized_data = quantize_nparray(dtype, data, scale, zero_point, low=quantize_range[0], high=quantize_range[1])
    return rmin, rmax, zero_point, scale, quantized_data


def qdq_data(data, qType, sym, reduce_range=False, ratio=1.0, axis=None):
    _, _, zero_point, scale, quantized_data = quantize_data(data, qType, sym, reduce_range, ratio, axis)
    return scale * (quantized_data - zero_point)


def is_B_transposed(node):
    """Whether inuput B is transposed."""
    transB = [attr for attr in node.attribute if attr.name == "transB"]
    if len(transB):
        return 0 < onnx.helper.get_attribute_value(transB[0])
    return False


def is_quantizable_type(data_type):
    return data_type in [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16]


def _get_blob_size(group_size, has_zp):  # pragma: no cover
    """Get blob_size.

    Args:
        group_size (int): how many elements share one scale/zp
        has_zp (bool): whether zero_point is None
    """
    if version.Version(ort.__version__) > constants.ONNXRT1161_VERSION:
        blob_size = group_size // 2
    elif has_zp:
        blob_size = group_size // 2 + 4 + 1
    else:
        blob_size = group_size // 2 + 4
    return blob_size


def make_weight_only_dequant_node(node, weight_shape, block_size, k_blocks, num_bits, q_weight, scale, zero_point, axis=1):
    """Build DequantizeLinear node.
    Args:
        node: original matmul node
        weight_shape (tuple): original weight shape
        block_size (int): how many elements share one scale/zp
        k_blocks (int): block number
        num_bits (int): num_bits
        q_weight (array): quantized weight
        scale (array): scale
        zero_point (array): zero point
        axis (int): the axis of the dequantizing dimension of the input tensor
    Returns:
        weight_only_dequant_node: DequantizeLinear node for weight dequantization
        new_inits: initializers of the new node
    """
    new_inits = []
    input_names = []
    kwargs = {
        "block_size": block_size,
        "axis": axis
        }

    q_weight_pairs = q_weight[::2, :] | q_weight[1::2, :] << 4

    q_weight_tensor = onnx.helper.make_tensor(
        name=node.input[1] + "_Q{}G{}".format(str(num_bits), str(block_size)),
        data_type=onnx.TensorProto.UINT4,
        dims=weight_shape,
        vals=q_weight_pairs.T.flatten().tobytes(),
        raw=True,
    )
    new_inits.append(q_weight_tensor)
    input_names.append(q_weight_tensor.name)

    #scale = scale.reshape((-1, weight_shape[-1]))
    scale_tensor = onnx.helper.make_tensor(
        name=node.input[1] + "_scale",
        data_type=onnx.helper.np_dtype_to_tensor_dtype(scale.dtype),
        dims=scale.shape,
        vals=scale.tobytes(),
        raw=True,
    )
    input_names.append(scale_tensor.name)
    new_inits.append(scale_tensor)

    # build zero_point tensor
    packed_zp = zero_point[:, ::2] | zero_point[:, 1::2] << 4

    zp_tensor = onnx.helper.make_tensor(
        name=node.input[1] + "_zp",
        data_type=onnx.TensorProto.UINT4,
        dims=scale.shape,
        vals=packed_zp.flatten().tobytes(),
        raw=True,
    )
    input_names.append(zp_tensor.name)
    new_inits.append(zp_tensor)

    dequant_node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=input_names,
        outputs=[q_weight_tensor.name + "_dequant"],
        name=node.name + "_woq_dequant",
        **kwargs,
    )
    node.input[1] = dequant_node.output[0]
    return dequant_node, new_inits


def make_matmul_weight_only_node(
    node: onnx.NodeProto,
    weight_shape: tuple,
    num_bits: int,
    group_size: int,
    k_blocks: int,
    q_weight: np.array,
    scale: np.array,
    zero_point: np.array,
    accuracy_level: int = 0,
):
    """Build MatMulFpQ4/MatMulNBits node.

    Args:
        node (onnx.NodeProto): original matmul node
        weight_shape (tuple): original weight shape
        num_bits (int): number of bits used to represent weights.
        group_size (int): how many elements share one scale/zp
        k_blocks (int): block number
        q_weight (np.array): quantized weight
        scale (np.array): scale
        zero_point (np.array): zero point
        accuracy_level (int, optional): accuracy level.
            Support 0 (unset), 1(fp32 compute type of jblas kernel),
            2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
            4 (int8 compute type of jblas kernel) Defaults to 0.

    Returns:
        matmul_weight_only_node: MatMulFpQ4 or MatMulNBits node
        new_inits: initializers of the new node
    """
    blob_size = _get_blob_size(group_size, zero_point is not None)
    packed = np.zeros((q_weight.shape[0], blob_size), dtype="uint8")
    q_weight_name = node.input[1] + "_Q{}G{}".format(str(num_bits), str(group_size))
    input_names = [node.input[0], q_weight_name]
    new_inits = []
    kwargs = {}

    if version.Version(ort.__version__) > constants.ONNXRT1161_VERSION:
        op_type = "MatMulNBits"

        # pack quantized weight
        q_weight_pairs = q_weight[:, ::2] | q_weight[:, 1::2] << 4
        packed[:, :] = q_weight_pairs[:, :blob_size]
        packed = np.reshape(packed, (-1, k_blocks, blob_size))

        # build scale tensor
        scale = np.reshape(scale, (-1, k_blocks))
        scale_tensor = onnx.helper.make_tensor(
            name=node.input[1] + "_scale",
            data_type=onnx.helper.np_dtype_to_tensor_dtype(scale.dtype),
            dims=scale.shape,
            vals=scale.tobytes(),
            raw=True,
        )
        input_names.append(scale_tensor.name)
        new_inits.append(scale_tensor)

        # build zero_point tensor
        if zero_point is not None:
            if num_bits > 4:
                packed_zp = np.reshape(zero_point, (1, -1)).astype("uint8")
            else:
                packed_zp = np.full((zero_point.shape[0] + 1) // 2, 136, dtype="uint8")
                # create an index array
                idx = np.arange(zero_point.shape[0] // k_blocks * k_blocks).reshape(-1)
                # separate odd and even indices
                even_idx = idx[::2]
                odd_idx = idx[1::2]
                # vectorized operation for even and odd indices
                packed_zp[even_idx // 2] = (packed_zp[even_idx // 2] & 0xF0) | zero_point[even_idx].ravel()
                packed_zp[odd_idx // 2] = (packed_zp[odd_idx // 2] & 0x0F) | (zero_point[odd_idx].ravel() << 4)

            zp_tensor = onnx.helper.make_tensor(
                name=node.input[1] + "_zp", data_type=2, dims=packed_zp.shape, vals=packed_zp.tobytes(), raw=True
            )
            input_names.append(zp_tensor.name)
            new_inits.append(zp_tensor)

        # set kwargs
        kwargs["K"] = weight_shape[0]
        kwargs["N"] = weight_shape[1]
        kwargs["bits"] = num_bits
        kwargs["block_size"] = group_size
        if accuracy_level > 0:
            # require onnxruntime > 1.16.3
            kwargs["accuracy_level"] = accuracy_level

    else:  # pragma: no cover
        offset = 5 if zero_point is not None else 4
        op_type = "MatMulFpQ4"

        # pack quantized weight
        for i in range(q_weight.shape[0]):
            bf = struct.pack("f", scale[i])
            packed[i][0] = bf[0]
            packed[i][1] = bf[1]
            packed[i][2] = bf[2]
            packed[i][3] = bf[3]

            if zero_point is not None:
                packed[i][4] = zero_point[i]

            packed[i][offset:] = np.bitwise_or(
                q_weight[i][: group_size // 2], np.left_shift(q_weight[i][group_size // 2 :], num_bits)
            )
        packed = packed.reshape(-1)

        # build shape tensor
        shape_tensor = onnx.helper.make_tensor(
            name=node.input[1] + "_shape", data_type=7, dims=(2,), vals=np.array(weight_shape, dtype="int64")
        )
        new_inits.append(shape_tensor)
        input_names.append(shape_tensor.name)

        # set kwargs
        kwargs["blk_quant_type"] = 1 if zero_point is not None else 0

    q_weight_tensor = onnx.helper.make_tensor(
        name=q_weight_name,
        data_type=2,
        dims=packed.shape,
        vals=packed.tobytes(),
        raw=True,
    )
    new_inits.append(q_weight_tensor)

    matmul_weight_only_node = onnx.helper.make_node(
        op_type,
        inputs=input_names,
        outputs=node.output,
        name=node.name + "_Q" + str(num_bits) if node.name else "_Q" + str(num_bits),
        domain="com.microsoft",
        **kwargs,
    )
    return matmul_weight_only_node, new_inits


def prepare_inputs(model, data_reader, providers):
    """Prepare inputs for weight only quantization.

    Args:
        model (ModelProto or onnx_model.ONNXModel): onnx model.
        data_reader (CalibrationDataReader): a calibration data reader.
        providers (list): providers to use.

    Returns:
        inputs: prepared inputs.
        so: session options
    """

    so = ort.SessionOptions()
    if sys.version_info < (3, 11) and util.find_spec("onnxruntime_extensions"):  # pragma: no cover
        so.register_custom_ops_library(onnxruntime_extensions.get_library_path())
    if model.is_large_model:
        onnx.save_model(
            model.model,
            model.model_path + "_augment.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            convert_attribute=False,
        )

    inputs_list = []
    while True:
        inputs = data_reader.get_next()
        if not inputs:
            break
        inputs_list.append(inputs)
    return inputs_list, so


def pad_tensor(weight, group_size, k_blocks):
    """Pad tensor rowi so that it can be is divisible by group_size.

    Args:
        weight (array): weight
        group_size (int): how many elements share one scale/zp
        k_blocks (int): the number of block

    Returns:
        weight: paded weight
    """
    if group_size == -1:
        return weight

    org_w_shape = weight.shape
    padded_rows = k_blocks * group_size
    pad_len = padded_rows - org_w_shape[0]

    if pad_len > 0:
        weight = np.pad(weight, ((0, pad_len), (0, 0)), "constant")

    return weight


def dump_woq_stats(model, quantize_config):
    res = {}

    dtype_set = set()
    for node in model.graph.node:
        if node.name.split("_Q")[0] not in quantize_config:
            continue
        if node.op_type in ["MatMulFpQ4", "MatMulNBits"]:
            optype = "MatMul"
        else:
            optype = node.op_type

        if optype not in res:
            res[optype] = {}
        if re.fullmatch("^.*_Q\d*G\d*", node.input[1]):
            search_out = re.search("_Q\d*", node.input[1])
            dtype = "A32W{}G{}".format(
                node.input[1][search_out.start() + 2 : search_out.end()], node.input[1][search_out.end() + 1 :]
            )
        else:
            dtype = "FP32"
        dtype_set.add(dtype)

        if dtype in res[optype]:
            res[optype][dtype] += 1
        else:
            res[optype][dtype] = 1

    dtype_list = list(dtype_set)
    for dtype in dtype_list:
        for optype in res.keys():
            if dtype not in res[optype]:
                res[optype][dtype] = 0

    # update stats format for dump.
    field_names = ["Op Type", "Total"]
    field_names.extend(dtype_list)
    output_data = []
    for op_type in res.keys():
        field_results = [op_type, sum(res[op_type].values())]
        field_results.extend([res[op_type][dtype] for dtype in dtype_list])
        output_data.append(field_results)

    utility.Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()


def get_node_original_name(node) -> str:
    """Get the original name of the given node."""
    node_name: str = node.name
    # TODO how to handle the unquantized node that has the `_quant` suffix, such as `conv_quant`?
    if node_name.endswith(QUANT_OP_NAME_SUFFIX):
        return node_name[: -len(QUANT_OP_NAME_SUFFIX)]
    else:
        # For unquantized nodes
        return node_name


def split_shared_bias(model):
    """Split shared tensor."""
    input_name_to_nodes = model.input_name_to_nodes()
    for input_name, node_list in input_name_to_nodes.items():
        if len(node_list) > 1 and input_name in [i.name for i in model.model.graph.initializer]:
            for node in node_list[1:]:
                if node.op_type not in ["Conv", "FusedConv"]:
                    continue
                if len(node.input) > 2 and node.input[2] == input_name:
                    new_input_name = node.input[2] + "_nc_split_" + node.name
                    new_input = onnx.helper.make_tensor(
                        new_input_name,
                        model.get_initializer(input_name).data_type,
                        model.get_initializer(input_name).dims,
                        model.get_initializer(input_name).raw_data,
                        True,
                    )
                    model.add_initializer(new_input)
                    node.input[2] = new_input_name
    return model


def remove_init_from_model_input(model):
    """Remove initializer from model input."""
    inputs = model.model.graph.input
    name_to_input = {}
    for inp in inputs:
        name_to_input[inp.name] = inp
    for initializer in model.model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])


class QuantizedValue:
    """Represents a linearly quantized value (input/output/initializer)."""

    def __init__(
        self,
        name,
        new_quantized_name,
        scale_name,
        zero_point_name,
        axis=None,
        qType=1,
    ):
        """Initialization.

        Args:
            name (string): tensor name
            new_quantized_name (string): quantized tensor name
            scale_name (string): scale name
            zero_point_name (string): zero point name
            axis (int, optional): quantized axis. Defaults to None.
            qType (int, optional): quantized data type. Defaults to 1 (uint8).
        """
        self.name = name
        self.q_name = new_quantized_name
        self.scale_name = scale_name
        self.zp_name = zero_point_name
        self.axis = axis
        self.qType = qType


class QuantizedInitializer:
    """Represents a linearly quantized weight input from ONNX operators."""

    def __init__(
        self,
        name,
        initializer,
        rmins,
        rmaxs,
        zero_points,
        scales,
        data=[],
        quantized_data=[],
        axis=None,
        qType=1,
    ):
        """Initialization.

        Args:
            name (string): initializer name
            initializer (onnx.onnx_ml_pb2.TensorProto): initializer
            rmins (list): list of min value
            rmaxs (list): list of max value
            zero_points (list): list of zero point
            scales (list): list of scale
            data (list, optional): array version of the initializer. Defaults to [].
            quantized_data (list, optional): quantized data. Defaults to [].
            axis (int, optional): quantized axis. Defaults to None.
            qType (int, optional): quantized data type. Defaults to 1 (uint8).
        """
        self.name = name
        self.initializer = initializer  # TensorProto initializer in ONNX graph
        self.rmins = rmins  # List of minimum range for each axis
        self.rmaxs = rmaxs  # List of maximum range for each axis
        # 1D tensor of zero points computed for each axis. scalar if axis is empty
        self.zero_points = zero_points
        self.scales = scales  # 1D tensor of scales computed for each axis. scalar if axis is empty
        self.data = data  # original data from initializer TensorProto
        self.quantized_data = quantized_data  # weight-packed data from data
        # Scalar to specify which dimension in the initializer to weight pack.
        self.axis = axis
        # If empty, single zero point and scales computed from a single rmin and rmax
        self.qType = qType


def dump_model_op_stats(model, quantize_config, fp32_op_list):
    qdq_ops = ["QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"]
    res = {}
    for op_type in fp32_op_list:
        res[op_type] = {"INT8": 0, "FP32": 0}
    for op_type in qdq_ops:
        res[op_type] = {"INT8": 0, "FP32": 0}

    for node in model.graph.node:
        if node.name.endswith("_quant"):
            if node.op_type.startswith("QLinear"):
                origin_op_type = node.op_type.split("QLinear")[-1]
            else:
                origin_op_type = node.op_type.split("Integer")[0]

            if origin_op_type in ["QAttention", "QGemm"]:
                origin_op_type = origin_op_type[1:]
            elif origin_op_type == "DynamicQuantizeLSTM":
                origin_op_type = "LSTM"
            elif origin_op_type == "QEmbedLayerNormalization":
                origin_op_type = "EmbedLayerNormalization"
            res[origin_op_type]["INT8"] += 1

        elif node.op_type in qdq_ops:
            res[node.op_type]["INT8"] += 1

        elif node.op_type in res:
            res[node.op_type]["FP32"] += 1

    field_names = ["Op Type", "Total", "INT8", "FP32"]
    output_data = [
        [
            op_type,
            sum(res[op_type].values()),
            res[op_type]["INT8"],
            res[op_type]["FP32"],
        ]
        for op_type in res.keys()
    ]

    utility.Statistics(output_data, header="Quantization Statistics", field_names=field_names).print_stat()
