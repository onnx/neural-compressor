#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import re
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Callable, List, NamedTuple, Union

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
from onnxruntime.quantization.quantize import DynamicQuantConfig as ORTDynamicQuantConfig
from onnxruntime.quantization.quantize import QuantConfig
from onnxruntime.quantization.quantize import StaticQuantConfig as ORTStaticQuantConfig

from neural_compressor_ort.quantization.calibrate import CalibrationDataReader
from neural_compressor_ort.utils import (
    AWQ,
    DEFAULT_WHITE_LIST,
    GPTQ,
    OP_NAME_OR_MODULE_TYPE,
    PRIORITY_AWQ,
    PRIORITY_GPTQ,
    PRIORITY_RTN,
    PRIORITY_SMOOTH_QUANT,
    RTN,
    SMOOTH_QUANT,
    logger,
)
from neural_compressor_ort.utils.base_config import BaseConfig, register_config, register_supported_configs

__all__ = [
    "RTNConfig",
    "get_default_rtn_config",
    "GPTQConfig",
    "get_default_gptq_config",
    "AWQConfig",
    "get_default_awq_config",
    "SmoothQuantConfig",
    "get_default_sq_config",
]


class _OperatorConfig(NamedTuple):
    config: BaseConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


######################## RNT Config ###############################


@register_config(algo_name=RTN, priority=PRIORITY_RTN)
class RTNConfig(BaseConfig):
    """Config class for round-to-nearest weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
        "ratios",
    ]
    model_params_list: List[str] = [
        "providers",
        "layer_wise_quant",
    ]
    name: str = RTN

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        ratios: dict = {},
        providers: List[str] = ["CPUExecutionProvider"],
        layer_wise_quant: bool = False,
        quant_last_matmul: bool = True,
        white_list: List[OP_NAME_OR_MODULE_TYPE] = DEFAULT_WHITE_LIST,
    ):
        """Init RTN weight-only quantization config.

        Args:
            weight_dtype (str, optional): Data type for weights, default is "int".
            weight_bits (int, optional): Number of bits used to represent weights, default is 4.
            weight_group_size (int, optional): Size of weight groups, default is 32.
            weight_sym (bool, optional): Indicates whether weights are symmetric, default is True.
            act_dtype (str, optional): Data type for activations, default is "fp32".
            accuracy_level (int, optional): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                4 (int8 compute type of jblas kernel). Defaults to 0.
            ratios (dict, optional): percentile of clip. Defaults to {}.
            providers (list, optional): execution providers to use. Defaults to ["CPUExecutionProvider"].
            layer_wise_quant (bool, optional): whether to quantize model layer by layer to save memory footprint.
                Check below link for details
                https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_layer_wise.md,
                default is False.
            quant_last_matmul (bool, optional): whether to quantize the last matmul of the model, default is True.
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.accuracy_level = accuracy_level
        self.ratios = ratios
        self.providers = providers
        self.layer_wise_quant = layer_wise_quant
        self.quant_last_matmul = quant_last_matmul
        self._post_init()

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    @classmethod
    def register_supported_configs(cls) -> List[_OperatorConfig]:
        supported_configs = []
        linear_rtn_config = RTNConfig(
            weight_dtype=["int"],
            weight_bits=[1, 2, 3, 4, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
        )
        operators = ["MatMul"]
        supported_configs.append(_OperatorConfig(config=linear_rtn_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(self, config_list: List[BaseConfig] = None, model_info: list = None):
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            config_mapping.update(config.get_model_params_dict())

            # update node level setting
            global_config = config.global_config
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if self.global_config is not None:
                    config_mapping[(op_name, op_type)] = global_config
                if op_type in op_type_config_dict:
                    config_mapping[(op_name, op_type)] = op_name_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = op_name_config_dict[op_name_pattern]
        if not self.quant_last_matmul:
            config_mapping[model_info[-1]] = {
                "weight": {"dtype": "fp32"},
                "activation": {"dtype": "fp32", "quant_mode": "fp32"},
            }
        return config_mapping

    @staticmethod
    def get_model_info(model: Union[onnx.ModelProto, Path, str]) -> list:
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model, load_external_data=False)
        white_list = ["MatMul"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "RTNConfig", List["RTNConfig"]]:  # pragma: no cover
        return RTNConfig(weight_bits=[4, 8], weight_sym=[True, False])


def get_default_rtn_config() -> RTNConfig:
    """Generate the default rtn config.

    Returns:
        the default rtn config.
    """
    return RTNConfig()


######################## GPTQ Config ###############################


@register_config(algo_name=GPTQ, priority=PRIORITY_GPTQ)
class GPTQConfig(BaseConfig):
    """Config class for gptq weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
    ]
    model_params_list: List[str] = [
        "percdamp",
        "blocksize",
        "actorder",
        "mse",
        "perchannel",
        "providers",
        "layer_wise_quant",
    ]
    name: str = GPTQ

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        percdamp: float = 0.01,
        blocksize: int = 128,
        actorder: bool = False,
        mse: bool = False,
        perchannel: bool = True,
        providers: List[str] = ["CPUExecutionProvider"],
        layer_wise_quant: bool = False,
        quant_last_matmul: bool = True,
        white_list: List[OP_NAME_OR_MODULE_TYPE] = DEFAULT_WHITE_LIST,
    ):
        """Init GPTQ weight-only quantization config.

        Args:
            weight_dtype (str, optional): data type for weights. Defaults to "int".
            weight_bits (int, optional): number of bits used to represent weights. Defaults to 4.
            weight_group_size (int, optional): size of weight groups. Defaults to 32.
            weight_sym (bool, optional): indicates whether weights are symmetric. Defaults to True.
            act_dtype (str, optional): data type for activations. Defaults to "fp32".
            accuracy_level (int, optional): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                4 (int8 compute type of jblas kernel). Defaults to 0.
            percdamp (float, optional): percentage of Hessian's diagonal values' average, which will be added
                to Hessian's diagonal to increase numerical stability. Defaults to 0.01.
            blocksize (int, optional): execute GPTQ quantization per block. Defaults to 128.
            actorder (bool, optional): whether to sort Hessian's diagonal values to rearrange channel-wise
                quantization order. Defaults to False.
            mse (bool, optional): whether get scale and zero point with mse error. Defaults to False.
            perchannel (bool, optional): whether quantize weight per-channel. Defaults to True.
            providers (list, optional): execution providers to use. Defaults to ["CPUExecutionProvider"].
            layer_wise_quant (bool, optional): whether to quantize model layer by layer to save memory footprint.
                Check below link for details
                https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_layer_wise.md,
                default is False.
            quant_last_matmul (bool, optional): whether to quantize the last matmul of the model, default is True.
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.accuracy_level = accuracy_level
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.actorder = actorder
        self.mse = mse
        self.perchannel = perchannel
        self.providers = providers
        self.layer_wise_quant = layer_wise_quant
        self.quant_last_matmul = quant_last_matmul
        self._post_init()

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    @classmethod
    def register_supported_configs(cls) -> List[_OperatorConfig]:
        supported_configs = []
        linear_gptq_config = GPTQConfig(
            weight_dtype=["int"],
            weight_bits=[1, 2, 3, 4, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
            actorder=[True, False],
            mse=[True, False],
            perchannel=[True, False],
        )
        operators = ["MatMul"]
        supported_configs.append(_OperatorConfig(config=linear_gptq_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(self, config_list: list = None, model_info: list = None) -> OrderedDict:
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            config_mapping.update(config.get_model_params_dict())

            # update node level setting
            global_config = config.global_config
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if self.global_config is not None:
                    config_mapping[(op_name, op_type)] = global_config
                if op_type in op_type_config_dict:
                    config_mapping[(op_name, op_type)] = op_name_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = op_name_config_dict[op_name_pattern]
        if not self.quant_last_matmul:
            config_mapping[model_info[-1]] = {
                "weight": {"dtype": "fp32"},
                "activation": {"dtype": "fp32", "quant_mode": "fp32"},
            }
        return config_mapping

    @staticmethod
    def get_model_info(model: Union[onnx.ModelProto, Path, str]) -> list:
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model, load_external_data=False)
        white_list = ["MatMul"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "GPTQConfig", List["GPTQConfig"]]:  # pragma: no cover
        return GPTQConfig(
            weight_bits=[4, 8],
            weight_sym=[True, False],
            actorder=[True, False],
            mse=[True, False],
            perchannel=[True, False],
        )


def get_default_gptq_config() -> GPTQConfig:
    """Generate the default gptq config.

    Returns:
        the default gptq config.
    """
    return GPTQConfig()


######################## AWQ Config ###############################


@register_config(algo_name=AWQ, priority=PRIORITY_AWQ)
class AWQConfig(BaseConfig):
    """Config class for awq weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
    ]
    model_params_list: List[str] = [
        "enable_auto_scale",
        "enable_mse_search",
        "providers",
    ]
    name: str = AWQ

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        enable_auto_scale: bool = True,
        enable_mse_search: bool = True,
        providers: List[str] = ["CPUExecutionProvider"],
        quant_last_matmul: bool = True,
        white_list: List[OP_NAME_OR_MODULE_TYPE] = DEFAULT_WHITE_LIST,
    ):
        """Init AWQ weight-only quantization config.

        Args:
            weight_dtype (str, optional): data type for weights. Defaults to "int".
            weight_bits (int, optional): number of bits used to represent weights. Defaults to 4.
            weight_group_size (int, optional): size of weight groups. Defaults to 32.
            weight_sym (bool, optional): indicates whether weights are symmetric. Defaults to True.
            act_dtype (str, optional): data type for activations. Defaults to "fp32".
            accuracy_level (int, optional): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                4 (int8 compute type of jblas kernel). Defaults to 0.
            enable_auto_scale (bool, optional): whether to search for best scales based on activation distribution.
                Defaults to True.
            enable_mse_search (bool, optional): whether to search for the best clip range from range
                [0.91, 1.0, 0.01]. Defaults to True.
            providers (list, optional): execution providers to use. Defaults to ["CPUExecutionProvider"].
            quant_last_matmul (bool, optional): whether to quantize the last matmul of the model, default is True.
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to DEFAULT_WHITE_LIST.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.accuracy_level = accuracy_level
        self.enable_auto_scale = enable_auto_scale
        self.enable_mse_search = enable_mse_search
        self.providers = providers
        self.quant_last_matmul = quant_last_matmul
        self._post_init()

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    @classmethod
    def register_supported_configs(cls) -> List[_OperatorConfig]:
        supported_configs = []
        linear_awq_config = AWQConfig(
            weight_dtype=["int"],
            weight_bits=[1, 2, 3, 4, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
            enable_auto_scale=[True, False],
            enable_mse_search=[True, False],
        )
        operators = ["MatMul"]
        supported_configs.append(_OperatorConfig(config=linear_awq_config, operators=operators))
        cls.supported_configs = supported_configs

    def to_config_mapping(self, config_list: list = None, model_info: list = None) -> OrderedDict:
        config_mapping = OrderedDict()
        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            config_mapping.update(config.get_model_params_dict())

            # update node level setting
            global_config = config.global_config
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if self.global_config is not None:
                    config_mapping[(op_name, op_type)] = global_config
                if op_type in op_type_config_dict:
                    config_mapping[(op_name, op_type)] = op_name_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        config_mapping[(op_name, op_type)] = op_name_config_dict[op_name_pattern]
        if not self.quant_last_matmul:
            config_mapping[model_info[-1]] = {
                "weight": {"dtype": "fp32"},
                "activation": {"dtype": "fp32", "quant_mode": "fp32"},
            }
        return config_mapping

    @staticmethod
    def get_model_info(model: Union[onnx.ModelProto, Path, str]) -> list:
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model, load_external_data=False)
        white_list = ["MatMul"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "AWQConfig", List["AWQConfig"]]:  # pragma: no cover
        return AWQConfig(
            weight_bits=[4, 8],
            weight_sym=[True, False],
            enable_auto_scale=[True, False],
            enable_mse_search=[True, False],
        )


def get_default_awq_config() -> AWQConfig:
    """Generate the default awq config.

    Returns:
        the default awq config.
    """
    return AWQConfig()


######################## SmoohQuant Config ###############################


@register_config(algo_name=SMOOTH_QUANT, priority=PRIORITY_SMOOTH_QUANT)
class SmoothQuantConfig(BaseConfig, ORTStaticQuantConfig):
    """Smooth quant quantization config."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        # smooth parameters
        "alpha",
        "folding",
        "auto_alpha_args",
        "calib_iter",
        "scales_per_op",
    ]
    name: str = SMOOTH_QUANT

    def __init__(
        self,
        alpha: float = 0.5,
        folding: bool = True,
        op_types: List[str] = ["Gemm", "Conv", "MatMul", "FusedConv"],
        calib_iter: int = 100,
        scales_per_op: bool = True,
        auto_alpha_args: dict = {"alpha_min": 0.3, "alpha_max": 0.7, "alpha_step": 0.05, "attn_method": "min"},
        providers: List[str] = ["CPUExecutionProvider"],
        white_list: List[OP_NAME_OR_MODULE_TYPE] = DEFAULT_WHITE_LIST,
        **kwargs,
    ):
        """Init smooth quant config.

        Args:
            alpha (float, optional): alpha value to balance the quantization difficulty of activation and weight.
                Defaults to 0.5.
            folding (bool, optional): whether fold those foldable Mul which are inserted for smooth quant.
                Defaults to True.
            op_types (list, optional): the op type to be smooth quantized.
                Defaults to ["Gemm", "Conv", "MatMul", "FusedConv"].
            calib_iter (int, optional): iteration num for calibration. Defaults to 100.
            scales_per_op (bool, optional): True, each op will have an individual scale, mainlyfor accuracy.
                False, ops with the same input will share a scale, mainly for performance. Defaults to True.
            auto_alpha_args (dict, optional): settings for alpha tuning.
                Defaults to {"alpha_min": 0.3, "alpha_max": 0.7, "alpha_step": 0.05, "attn_method": "min"}.
            providers (list, optional): providers used for inference.
                Defaults to ["CPUExecutionProvider"].
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to DEFAULT_WHITE_LIST.
            kwargs (dict): kwargs in below link are supported except calibration_data_reader:
                https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py#L78
        """
        BaseConfig.__init__(self)
        kwargs.update({"calibration_data_reader": None})
        ORTStaticQuantConfig.__init__(self, **kwargs)
        self.alpha = alpha
        self.folding = folding
        self.op_types = op_types
        self.calib_iter = calib_iter
        self.scales_per_op = scales_per_op
        self.auto_alpha_args = auto_alpha_args
        self.providers = providers
        self.white_list = white_list
        self.weight_type = self.weight_type.value if isinstance(self.weight_type, Enum) else self.weight_type
        self.activation_type = (
            self.activation_type.value if isinstance(self.activation_type, Enum) else self.activation_type
        )
        self.calibrate_method = (
            self.calibrate_method.value if isinstance(self.calibrate_method, Enum) else self.calibrate_method
        )
        self.quant_format = self.quant_format.value if isinstance(self.quant_format, Enum) else self.quant_format
        self._post_init()

    @classmethod
    def register_supported_configs(cls) -> List[_OperatorConfig]:
        supported_configs = []
        smooth_quant_config = SmoothQuantConfig()
        operators = ["Gemm", "Conv", "MatMul", "FusedConv"]
        supported_configs.append(_OperatorConfig(config=smooth_quant_config, operators=operators))
        cls.supported_configs = supported_configs

    @staticmethod
    def get_model_info(model) -> list:
        white_list = ["Gemm", "Conv", "MatMul", "FusedConv"]
        filter_result = []
        for node in model.graph.node:
            if node.op_type in white_list:
                pair = (node.name, node.op_type)
                filter_result.append(pair)
        logger.debug(f"Get model info: {filter_result}")
        return filter_result

    @classmethod
    def get_config_set_for_tuning(
        cls,
    ) -> Union[None, "SmoothQuantConfig", List["SmoothQuantConfig"]]:  # pragma: no cover
        return SmoothQuantConfig(alpha=np.arange(0.3, 0.7, 0.05))

    def convert_to_ort_config(self):
        self.activation_type = QuantType(self.activation_type)
        self.weight_type = QuantType(self.weight_type)
        self.weight_type = QuantType(self.weight_type)
        self.calibrate_method = CalibrationMethod(self.calibrate_method)
        self.quant_format = QuantFormat(self.quant_format)


def get_default_sq_config() -> SmoothQuantConfig:
    """Generate the default smooth quant config.

    Returns:
        the default smooth quant config.
    """
    return SmoothQuantConfig()


######################## WOQ Tuning Config ###############################


def get_woq_tuning_config() -> list:
    """Generate the config set for WOQ tuning.

    Returns:
        the list of WOQ quant config.
    """
    RTN_G32ASYM = RTNConfig(weight_sym=False)
    GPTQ_G32ASYM = GPTQConfig(weight_sym=False)
    GPTQ_G32ASYM_DISABLE_LAST_MATMUL = GPTQConfig(weight_sym=False, quant_last_matmul=False)
    GPTQ_G128ASYM = GPTQConfig(weight_group_size=128, weight_sym=False)
    AWQ_G32ASYM = AWQConfig(weight_sym=False)
    return [RTN_G32ASYM, GPTQ_G32ASYM, GPTQ_G32ASYM_DISABLE_LAST_MATMUL, GPTQ_G128ASYM, AWQ_G32ASYM]


##################### INC Algo Configs End ###################################


register_supported_configs()


##################### Config for ONNXRuntime-like user-facing API ############


class StaticQuantConfig(ORTStaticQuantConfig):
    def __init__(self, calibration_data_reader: CalibrationDataReader, extra_options=None, *args, **kwargs):
        """This is a class for static Quant Configuration.

        Inherit from StaticQuantConfig:
        https://github.com/microsoft/onnxruntime/blob/v1.17.1/onnxruntime/python/tools/quantization/quantize.py#L78
        extra_options:
            Support smoothquant args.
            - SmoothQuant = True/False :
                Default is False. If enabled, SmoothQuant algorithm will be applied before quantization to do
                fake input channel quantization.
            - SmoothQuantAlpha = float :
                Default is 0.5. It only works if SmoothQuant is True. It controls the difficulty of weight
                and activation quantization. A larger alpha value could be used on models with more significant
                activation outliers to migrate more quantization difficulty to weights.
            - SmoothQuantFolding = True/False :
                Default is True. It only works if SmoothQuant is True. If enabled, inserted Mul ops during
                SmoothQuant will be folded into the previous op if the previous op is foldable.
            - SmoothQuantOpTypes = list (new args):
                Default is ["Gemm", "Conv", "MatMul", "FusedConv"]. It only works if SmoothQuant is True.
                It controls the op types to be smooth quantized.
            - SmoothQuantCalibIter = int (new args):
                Default is 100. It only works if SmoothQuant is True. It controls the iteration num for calibration.
            - SmoothQuantScalesPerOp = True/False (new args) :
                Default is True. It only works if SmoothQuant is True.
                If enabled, each op will have an individual scale, mainlyfor accuracy.
                If not enabled,  ops with the same input will share a scale, mainly for performance.
        """
        super().__init__(calibration_data_reader=calibration_data_reader, extra_options=extra_options, *args, **kwargs)

    def to_dict(self):
        return self.__dict__


class DynamicQuantConfig(ORTDynamicQuantConfig):
    """This is a class for dynamic Quant Configuration.

    Inherit from DynamicQuantConfig:
        https://github.com/microsoft/onnxruntime/blob/v1.17.1/onnxruntime/python/tools/quantization/quantize.py#L206
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def generate_inc_sq_config(quant_config: QuantConfig):
    extra_options = quant_config.extra_options
    quant_kwargs = {
        "alpha": extra_options.get("SmoothQuantAlpha", 0.5),
        "folding": extra_options.get("SmoothQuantFolding", True),
        "op_types": extra_options.get("SmoothQuantOpTypes", ["Gemm", "Conv", "MatMul", "FusedConv"]),
        "calib_iter": extra_options.get("SmoothQuantCalibIter", 100),
        "scales_per_op": extra_options.get("SmoothQuantScalesPerOp", True),
    }
    quant_config.extra_options["SmoothQuant"] = False
    quant_config_dict = quant_config.to_dict()
    inc_sq_config = SmoothQuantConfig(**quant_kwargs, **quant_config_dict)
    return inc_sq_config
