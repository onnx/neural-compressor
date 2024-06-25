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

import pathlib
import tempfile
from typing import Union

import onnx
import onnxruntime as ort
from onnxruntime.quantization.quantize import QuantConfig

from onnx_neural_compressor.quantization import algorithm_entry as algos
from onnx_neural_compressor.quantization import config


# ORT-like user-facing API
def quantize(
    model_input: Union[str, pathlib.Path, onnx.ModelProto],
    model_output: Union[str, pathlib.Path],
    quant_config: config.BaseConfig,
    optimization_level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
):
    with tempfile.TemporaryDirectory(prefix="ort.opt.") as tmp_dir:
        if optimization_level != ort.GraphOptimizationLevel.ORT_DISABLE_ALL:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = optimization_level
            sess_options.optimized_model_filepath = pathlib.Path(tmp_dir).joinpath("opt.onnx").as_posix()
            session = ort.InferenceSession(model_input, sess_options)
            del session
            model_input = sess_options.optimized_model_filepath

        if isinstance(quant_config, config.StaticQuantConfig):
            if quant_config.extra_options.get("SmoothQuant", False):
                algos.smooth_quant_entry(
                    model_input, quant_config, quant_config.calibration_data_reader, model_output=model_output
                )
            else:
                algos.static_quantize_entry(
                    model_input, quant_config, quant_config.calibration_data_reader, model_output=model_output
                )
        elif isinstance(quant_config, config.DynamicQuantConfig):
            algos.dynamic_quantize_entry(model_input, quant_config, model_output=model_output)
        else:
            raise TypeError(
                "Invalid quantization config type, it must be either StaticQuantConfig or DynamicQuantConfig."
            )
