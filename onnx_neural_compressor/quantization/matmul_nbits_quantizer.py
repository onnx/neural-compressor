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

from typing import List, Union  # isort: skip

import pathlib
import tempfile

import onnx
import onnxruntime as ort

from onnx_neural_compressor import data_reader, logger, onnx_model, utility
from onnx_neural_compressor.quantization import algorithm_entry as algos
from onnx_neural_compressor.quantization import config, QuantFormat


class WeightOnlyQuantConfig:
    def __init__(self, algorithm, quant_format=QuantFormat.QOperator):
        """This is the Base class for Weight Only Quant Configuration.

        Args:
            algorithm:
                weight only quantize algorithm name.
        """
        self.algorithm = algorithm
        self.quant_format = quant_format


class RTNWeightOnlyQuantConfig(WeightOnlyQuantConfig):

    def __init__(self, ratios=None, layer_wise_quant=False, quant_format=QuantFormat.QOperator):
        super().__init__(
            algorithm="RTN",
            quant_format=quant_format,
        )
        if ratios is None:
            ratios = {}
        self.ratios = ratios
        self.layer_wise_quant = layer_wise_quant


class GPTQWeightOnlyQuantConfig(WeightOnlyQuantConfig):

    def __init__(
        self,
        calibration_data_reader: data_reader.CalibrationDataReader,
        percdamp=0.01,
        block_size=128,
        actorder=False,
        mse=False,
        perchannel=True,
        layer_wise_quant=False,
        quant_format=QuantFormat.QOperator,
    ):
        super().__init__(
            algorithm="GPTQ",
            quant_format=quant_format,
        )
        self.calibration_data_reader = calibration_data_reader
        self.percdamp = percdamp
        self.block_size = block_size
        self.actorder = actorder
        self.mse = mse
        self.perchannel = perchannel
        self.layer_wise_quant = layer_wise_quant


class AWQWeightOnlyQuantConfig(WeightOnlyQuantConfig):

    def __init__(
        self,
        calibration_data_reader: data_reader.CalibrationDataReader,
        enable_auto_scale=True,
        enable_mse_search=True,
        quant_format=QuantFormat.QOperator,
    ):
        super().__init__(algorithm="AWQ", quant_format=quant_format)
        self.calibration_data_reader = calibration_data_reader
        self.enable_auto_scale = enable_auto_scale
        self.enable_mse_search = enable_mse_search


algorithm_config_mapping = {
    "RTN": RTNWeightOnlyQuantConfig,
    "AWQ": AWQWeightOnlyQuantConfig,
    "GPTQ": GPTQWeightOnlyQuantConfig,
}


class MatMulNBitsQuantizer:

    def __init__(
        self,
        model: Union[onnx.ModelProto, str],
        block_size: int = 128,
        is_symmetric: bool = False,
        accuracy_level: int = 0,
        nodes_to_exclude: List[str] = None,
        algo_config: WeightOnlyQuantConfig = None,
        n_bits: int = 4,
        providers: List[str] = ["CPUExecutionProvider"],
        optimization_level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    ):
        if nodes_to_exclude is None:
            nodes_to_exclude = []
        self.model = model
        self.block_size = block_size
        self.is_symmetric = is_symmetric
        self.accuracy_level = accuracy_level
        self.nodes_to_exclude = list(set(nodes_to_exclude))
        self.algo_config = algo_config or RTNWeightOnlyQuantConfig()
        self.n_bits = n_bits
        self.providers = providers
        self.algorithm = self.algo_config.algorithm
        self.optimization_level = optimization_level
        assert self.algorithm in [
            "RTN",
            "AWQ",
            "GPTQ",
        ], "Only RTN, GPTQ and AWQ algorithms are supported, but get {} algorithm".format(self.algorithm)

    def _generate_nc_config(self):
        config_class = config.config_registry.get_cls_configs()[self.algorithm.lower()]
        quant_kwargs = {
            "weight_bits": self.n_bits,
            "weight_group_size": self.block_size,
            "weight_sym": self.is_symmetric,
            "accuracy_level": self.accuracy_level,
            "providers": self.providers,
            "quant_format": self.algo_config.quant_format,
        }
        if self.algorithm == "RTN":
            quant_kwargs.update(
                {
                    "layer_wise_quant": self.algo_config.layer_wise_quant,
                }
            )
        elif self.algorithm == "GPTQ":
            quant_kwargs.update(
                {
                    "percdamp": self.algo_config.percdamp,
                    "block_size": self.algo_config.block_size,
                    "actorder": self.algo_config.actorder,
                    "mse": self.algo_config.mse,
                    "perchannel": self.algo_config.perchannel,
                    "layer_wise_quant": self.algo_config.layer_wise_quant,
                }
            )
        elif self.algorithm == "AWQ":
            quant_kwargs.update(
                {
                    "enable_auto_scale": self.algo_config.enable_auto_scale,
                    "enable_mse_search": self.algo_config.enable_mse_search,
                }
            )
        nc_config = config_class(**quant_kwargs)

        if len(self.nodes_to_exclude) > 0:
            not_quant_kwargs = {"weight_dtype": "fp32", "white_list": self.nodes_to_exclude}
            nc_config += config_class(**not_quant_kwargs)

        return nc_config

    def int4_quant_algo(self):
        qconfig = self._generate_nc_config()
        model = self.model
        opt_tmp_file = tempfile.TemporaryDirectory()

        if getattr(self.algo_config, "layer_wise_quant", False) and not isinstance(model, str):
            logger.warning("Please use model path for layer-wise quantization.")

        # do graph optimization if not layer_wise_quant
        if (
            not getattr(self.algo_config, "layer_wise_quant", False)
            and self.optimization_level != ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ):
            if not isinstance(model, str):
                onnx.save_model(
                    model,
                    pathlib.Path(opt_tmp_file.name).joinpath("tmp.onnx").as_posix(),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="tmp.onnx_data",
                    size_threshold=1024,
                    convert_attribute=False,
                )
                model = pathlib.Path(opt_tmp_file.name).joinpath("tmp.onnx").as_posix()
            logger.info("Start graph optimization...")
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = self.optimization_level
            sess_options.optimized_model_filepath = pathlib.Path(opt_tmp_file.name).joinpath("opt.onnx").as_posix()
            sess_options.add_session_config_entry(
                "session.optimized_model_external_initializers_file_name", "opt.onnx_data"
            )
            sess_options.add_session_config_entry(
                "session.optimized_model_external_initializers_min_size_in_bytes", "1024"
            )
            session = ort.InferenceSession(model, sess_options, providers=["CPUExecutionProvider"])
            model = sess_options.optimized_model_filepath
            del session
            logger.info("Graph optimization done.")

        logger.info(f"start to quantize model with {self.algorithm} algorithm...")
        if self.algorithm == "RTN":
            self.model = algos.rtn_quantize_entry(model, qconfig)
        elif self.algorithm == "GPTQ":
            self.model = algos.gptq_quantize_entry(model, qconfig, self.algo_config.calibration_data_reader)
        elif self.algorithm == "AWQ":
            self.model = algos.awq_quantize_entry(model, qconfig, self.algo_config.calibration_data_reader)
        logger.info(f"complete quantization of model with {self.algorithm} algorithm.")
        opt_tmp_file.cleanup()

    def process(self):
        self.int4_quant_algo()
