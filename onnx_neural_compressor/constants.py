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
"""Class for All constants."""

import datetime

from packaging import version

# constants for configs
GLOBAL = "global"
LOCAL = "local"
DEFAULT_WHITE_LIST = "*"
EMPTY_WHITE_LIST = None

# config name
BASE_CONFIG = "base_config"
COMPOSABLE_CONFIG = "composable_config"
RTN = "rtn"
STATIC_QUANT = "static_quant"
DYNAMIC_QUANT = "dynamic_quant"
SMOOTH_QUANT = "smooth_quant"
GPTQ = "gptq"
AWQ = "awq"

DEFAULT_WORKSPACE = "./nc_workspace/{}/".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


ONNXRT116_VERSION = version.Version("1.16.0")
ONNXRT1161_VERSION = version.Version("1.16.1")

PRIORITY_RTN = 60
PRIORITY_GPTQ = 70
PRIORITY_AWQ = 50
PRIORITY_SMOOTH_QUANT = 80
PRIORITY_STATIC_QUANT = 70
PRIORITY_DYNAMIC_QUANT = 60

MAXIMUM_PROTOBUF = 2147483648

WHITE_MODULE_LIST = ["MatMul", "Conv"]

RTN_OP_LIST = ["MatMul"]

AWQ_OP_LIST = ["MatMul"]

GPTQ_OP_LIST = ["MatMul"]

DYNAMIC_CPU_OP_LIST = ["FusedConv", "Conv", "EmbedLayerNormalization", "MatMul", "Gather", "Attention", "LSTM"]
DYNAMIC_CUDA_OP_LIST = ["FusedConv", "Conv", "EmbedLayerNormalization", "MatMul", "Gather", "Attention", "LSTM"]
DYNAMIC_DML_OP_LIST = []
DYNAMIC_DNNL_OP_LIST = ["FusedConv", "Conv", "EmbedLayerNormalization", "MatMul", "Gather", "Attention", "LSTM"]
DYNAMIC_TRT_OP_LIST = []

STATIC_QDQ_CPU_OP_LIST = [
    "FusedConv",
    "Conv",
    "Gather",
    "GatherElements",
    "GatherND",
    "Tile",
    "MatMul",
    "Gemm",
    "EmbedLayerNormalization",
    "Attention",
    "Relu",
    "Clip",
    "LeakyRelu",
    "Sigmoid",
    "MaxPool",
    "GlobalAveragePool",
    "Pad",
    "Split",
    "Squeeze",
    "Reshape",
    "Concat",
    "AveragePool",
    "Unsqueeze",
    "Transpose",
    "Resize",
    "Abs",
    "Shrink",
    "Sign",
    "Flatten",
    "Expand",
    "Slice",
    "Mod",
    "ReduceMax",
    "ReduceMin",
    "CenterCropPad",
]
STATIC_QDQ_CUDA_OP_LIST = [
    "FusedConv",
    "Conv",
    "Gather",
    "MatMul",
    "Gemm",
    "EmbedLayerNormalization",
    "Attention",
    "Relu",
    "Clip",
    "LeakyRelu",
    "Sigmoid",
    "MaxPool",
    "GlobalAveragePool",
    "Pad",
    "Split",
    "Squeeze",
    "Reshape",
    "Concat",
    "AveragePool",
    "Unsqueeze",
    "Transpose",
    "Resize",
    "Abs",
    "Shrink",
    "Sign",
    "Flatten",
    "Expand",
    "Slice",
    "Mod",
    "ReduceMax",
    "ReduceMin",
]
STATIC_QDQ_DML_OP_LIST = [
    "Conv",
    "MatMul",
    "Relu",
    "Clip",
    "MaxPool",
]
STATIC_QDQ_DNNL_OP_LIST = [
    "FusedConv",
    "Conv",
    "Gather",
    "MatMul",
    "Gemm",
    "EmbedLayerNormalization",
    "Attention",
    "Relu",
    "Clip",
    "LeakyRelu",
    "Sigmoid",
    "MaxPool",
    "GlobalAveragePool",
    "Pad",
    "Split",
    "Squeeze",
    "Reshape",
    "Concat",
    "AveragePool",
    "Unsqueeze",
    "Transpose",
    "Resize",
]
STATIC_QDQ_TRT_OP_LIST = [
    "Conv",
    "MatMul",
    "Attention",
    "LeakyRelu",
    "Gather",
    "Sigmoid",
    "MaxPool",
    "EmbedLayerNormalization",
    "GlobalAveragePool",
    "Pad",
    "Split",
    "Squeeze",
    "Reshape",
    "Concat",
    "AveragePool",
    "Unsqueeze",
    "Transpose",
    "Resize",
    "Gemm",
    "Add",
]

STATIC_QOPERATOR_CPU_OP_LIST = [
    "FusedConv",
    "Conv",
    "Gather",
    "GatherElements",
    "GatherND",
    "Tile",
    "MatMul",
    "Gemm",
    "EmbedLayerNormalization",
    "Attention",
    "Mul",
    "Relu",
    "Clip",
    "LeakyRelu",
    "Sigmoid",
    "MaxPool",
    "GlobalAveragePool",
    "Pad",
    "Split",
    "Add",
    "Squeeze",
    "Reshape",
    "Concat",
    "AveragePool",
    "Unsqueeze",
    "Transpose",
    "ArgMax",
    "Resize",
    "Abs",
    "Shrink",
    "Sign",
    "Flatten",
    "Expand",
    "Slice",
    "Mod",
    "ReduceMax",
    "ReduceMin",
    "CenterCropPad",
]
STATIC_QOPERATOR_CUDA_OP_LIST = [
    "FusedConv",
    "Conv",
    "Gather",
    "MatMul",
    "Gemm",
    "EmbedLayerNormalization",
    "Attention",
    "Mul",
    "Relu",
    "Clip",
    "LeakyRelu",
    "Sigmoid",
    "MaxPool",
    "GlobalAveragePool",
    "Pad",
    "Split",
    "Add",
    "Squeeze",
    "Reshape",
    "Concat",
    "AveragePool",
    "Unsqueeze",
    "Transpose",
    "ArgMax",
    "Resize",
    "Abs",
    "Shrink",
    "Sign",
    "Flatten",
    "Expand",
    "Slice",
    "Mod",
    "ReduceMax",
    "ReduceMin",
]
STATIC_QOPERATOR_DML_OP_LIST = [
    "Conv",
    "MatMul",
    "Mul",
    "Relu",
    "Clip",
    "MaxPool",
    "Add",
]
STATIC_QOPERATOR_DNNL_OP_LIST = [
    "FusedConv",
    "Conv",
    "Gather",
    "MatMul",
    "Gemm",
    "EmbedLayerNormalization",
    "Attention",
    "Mul",
    "Relu",
    "Clip",
    "LeakyRelu",
    "Sigmoid",
    "MaxPool",
    "GlobalAveragePool",
    "Pad",
    "Split",
    "Add",
    "Squeeze",
    "Reshape",
    "Concat",
    "AveragePool",
    "Unsqueeze",
    "Transpose",
    "ArgMax",
    "Resize",
]
STATIC_QOPERATOR_TRT_OP_LIST = []

STATIC_QOPERATOR_OP_LIST_MAP = {
    "CPUExecutionProvider": STATIC_QOPERATOR_CPU_OP_LIST,
    "CUDAExecutionProvider": STATIC_QOPERATOR_CUDA_OP_LIST,
    "DmlExecutionProvider": STATIC_QOPERATOR_DML_OP_LIST,
    "DnnlExecutionProvider": STATIC_QOPERATOR_DNNL_OP_LIST,
    "TensorrtExecutionProvider": STATIC_QOPERATOR_TRT_OP_LIST,
}

STATIC_QDQ_OP_LIST_MAP = {
    "CPUExecutionProvider": STATIC_QDQ_CPU_OP_LIST,
    "CUDAExecutionProvider": STATIC_QDQ_CUDA_OP_LIST,
    "DmlExecutionProvider": STATIC_QDQ_DML_OP_LIST,
    "DnnlExecutionProvider": STATIC_QDQ_DNNL_OP_LIST,
    "TensorrtExecutionProvider": STATIC_QDQ_TRT_OP_LIST,
}

DYNAMIC_OP_LIST_MAP = {
    "CPUExecutionProvider": DYNAMIC_CPU_OP_LIST,
    "CUDAExecutionProvider": DYNAMIC_CUDA_OP_LIST,
    "DmlExecutionProvider": DYNAMIC_DML_OP_LIST,
    "DnnlExecutionProvider": DYNAMIC_DNNL_OP_LIST,
    "TensorrtExecutionProvider": DYNAMIC_TRT_OP_LIST,
}
