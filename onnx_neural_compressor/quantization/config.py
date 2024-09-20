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

from __future__ import annotations

import copy
import dataclasses
import enum
import inspect
import itertools
import json
import os
import pathlib
import re
from abc import ABC, abstractmethod

import numpy as np
import onnx
import onnxruntime as ort
import pydantic
from onnxruntime import quantization as ort_quant
from packaging import version
from typing_extensions import Self

from onnx_neural_compressor import constants, data_reader, logger, quantization, utility

from collections import OrderedDict  # isort: skip
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union, _GenericAlias  # isort: skip

ort_version = version.Version(ort.__version__)


class ParamLevel(enum.Enum):
    OP_LEVEL = enum.auto()
    OP_TYPE_LEVEL = enum.auto()
    MODEL_LEVEL = enum.auto()


class TuningParam:
    """Define the tunable parameter for the algorithm.

    Example:
        Class FakeAlgoConfig(config.BaseConfig):
            '''Fake algo config.'''.

            params_list = [
                ...
                # For simple tunable types, like a list of int, giving
                # the param name is enough. `config.BaseConfig` class will
                # create the `TuningParam` implicitly.
                "simple_attr"

                # For complex tunable types, like a list of lists,
                # developers need to create the `TuningParam` explicitly.
                TuningParam("complex_attr", tunable_type=List[List[str]])

                # The default parameter level is `ParamLevel.OP_LEVEL`.
                # If the parameter is at a different level, developers need
                # to specify it explicitly.
                TuningParam("model_attr", level=ParamLevel.MODEL_LEVEL)

            ...

    # TODO: more examples to explain the usage of `TuningParam`.
    """

    def __init__(
        self,
        name: str,
        default_val: Any = None,
        tunable_type=None,
        options=None,
        level: ParamLevel = ParamLevel.OP_LEVEL,
    ) -> None:
        self.name = name
        self.default_val = default_val
        self.tunable_type = tunable_type
        self.options = options
        self.level = level

    @staticmethod
    def create_input_args_model(expect_args_type: Any) -> type:
        """Dynamically create an InputArgsModel based on the provided type hint.

        Parameters:
        - expect_args_type (Any): The user-provided type hint for input_args.

        Returns:
        - type: The dynamically created InputArgsModel class.
        """

        class DynamicInputArgsModel(pydantic.BaseModel):
            input_args: expect_args_type

        return DynamicInputArgsModel

    def is_tunable(self, value: Any) -> bool:
        # Use `Pydantic` to validate the input_args.
        # TODO: refine the implementation in further.
        assert isinstance(self.tunable_type, _GenericAlias), f"Expected a type hint, got {self.tunable_type} instead."
        DynamicInputArgsModel = TuningParam.create_input_args_model(self.tunable_type)
        try:
            new_args = DynamicInputArgsModel(input_args=value)
            return True
        except Exception as e:
            logger.debug(f"Failed to validate the input_args: {e}")
            return False

    def __str__(self) -> str:
        return "TuningParam(name={}, tunable_type={}, options={}).".format(
            self.name, str(self.tunable_type), str(self.options)
        )


# Config registry to store all registered configs.
class ConfigRegistry(object):
    registered_configs = {}
    _config_registry = None

    def __new__(cls) -> Self:
        if cls._config_registry is None:
            cls._config_registry = super(ConfigRegistry, cls).__new__(cls)

        return cls._config_registry

    @classmethod
    def register_config_impl(cls, algo_name: str, priority: Union[float, int] = 0):
        """Register config decorator.

        The register the configuration classes for different algorithms.

        Usage example:
            @ConfigRegistry.register_config(algo_name=ExampleAlgorithm, priority=100)
            class ExampleAlgorithmConfig:
                # Configuration details for the ExampleAlgorithm

        Args:
            algo_name: the algorithm name.
            priority: priority: the priority of the configuration. A larger number indicates a higher priority,
                which will be tried first at the auto-tune stage. Defaults to 0.
        """

        def decorator(config_cls):
            cls.registered_configs[algo_name] = {"priority": priority, "cls": config_cls}
            return config_cls

        return decorator

    @classmethod
    def get_all_configs(cls) -> Dict[str, Dict[str, Dict[str, object]]]:
        """Get all registered configurations."""
        return cls.registered_configs

    @classmethod
    def get_sorted_configs(cls) -> Dict[str, OrderedDict[str, Dict[str, object]]]:
        """Get registered configurations sorted by priority."""
        return OrderedDict(sorted(cls.registered_configs.items(), key=lambda x: x[1]["priority"], reverse=True))

    @classmethod
    def get_cls_configs(cls) -> Dict[str, Dict[str, object]]:
        """Get registered configurations without priority."""
        cls_configs = {}
        for algo_name, config_data in cls.registered_configs.items():
            cls_configs[algo_name] = config_data["cls"]
        return cls_configs

    @classmethod
    def get_all_config_cls(cls) -> List[Type[BaseConfig]]:
        configs_cls = []
        for algo_name, config_pairs in cls.registered_configs.items():
            configs_cls.append(config_pairs["cls"])
        return configs_cls


config_registry = ConfigRegistry()


def register_config(algo_name: str, priority: Union[float, int] = 0):
    """Register config decorator.

    The register the configuration classes for different algorithms.

    Usage example:
        @register_config(algo_name=ExampleAlgorithm, priority=100)
        class ExampleAlgorithmConfig:
            # Configuration details for the ExampleAlgorithm

    Args:
        algo_name: the algorithm name.
        priority: the priority of the configuration. A larger number indicates a higher priority,
            which will be tried first at the auto-tune stage. Defaults to 0.
    """

    return config_registry.register_config_impl(algo_name=algo_name, priority=priority)


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, quantization.QuantType):
            return getattr(o, "tensor_type")
        if isinstance(o, quantization.QuantFormat):
            return getattr(o, "value")
        if isinstance(o, quantization.CalibrationMethod):
            return getattr(o, "name")
        return super().default(o)


class BaseConfig(ABC):
    """The base config for all algorithm configs."""

    name = constants.BASE_CONFIG
    params_list: List[Union[str, TuningParam]] = []
    model_params_list: List[Union[str, TuningParam]] = []

    def __init__(
        self,
        white_list: Optional[List[str]] = constants.DEFAULT_WHITE_LIST,
    ) -> None:
        self._global_config: Optional[BaseConfig] = None
        # local config is the collections of operator_type configs and operator configs
        self._local_config: Dict[str, Optional[BaseConfig]] = {}
        self._white_list = white_list
        self._config_mapping = OrderedDict()
        self._post_init()

    def _post_init(self):
        if self.white_list == constants.DEFAULT_WHITE_LIST:
            global_config = self.get_init_args()
            self._global_config = self.__class__(**global_config, white_list=constants.EMPTY_WHITE_LIST)
        elif isinstance(self.white_list, list) and len(self.white_list) > 0:
            for op_name_or_type in self.white_list:
                global_config = self.get_init_args()
                tmp_config = self.__class__(**global_config, white_list=constants.EMPTY_WHITE_LIST)
                self.set_local(op_name_or_type, tmp_config)
        elif self.white_list == constants.EMPTY_WHITE_LIST:
            return
        else:
            raise NotImplementedError(
                f"The white list should be one of {constants.DEFAULT_WHITE_LIST}, {constants.EMPTY_WHITE_LIST},"
                " a not empty list, but got {self.white_list}"
            )

    @property
    def config_mapping(self):
        return self._config_mapping

    @property
    def white_list(self):
        return self._white_list

    @white_list.setter
    def white_list(self, op_name_or_type_list: Optional[List[Union[str, Callable]]]):
        self._white_list = op_name_or_type_list

    @property
    def global_config(self):
        return self._global_config

    @global_config.setter
    def global_config(self, config):
        self._global_config = config

    @property
    def local_config(self):
        return self._local_config

    @local_config.setter
    def local_config(self, config):
        self._local_config = config

    def set_local(self, operator_name: str, config: BaseConfig) -> BaseConfig:
        if operator_name in self.local_config and config != self.local_config[operator_name]:
            logger.debug("The configuration for %s has already been set, update it.", operator_name)
        self.local_config[operator_name] = config
        return self

    def to_dict(self):
        result = {}
        global_config = self.get_init_args()
        if bool(self.local_config):
            result[constants.LOCAL] = {}
            for op_name, config in self.local_config.items():
                result[constants.LOCAL][op_name] = config.to_dict()
            if global_config:
                result[constants.GLOBAL] = global_config
        else:
            result = global_config
        return result

    def get_params_dict(self):
        result = dict()
        for param, value in self.__dict__.items():
            if param in self.params_list:
                result[param] = value
        return result

    def get_init_args(self):
        result = dict()
        for param, value in self.__dict__.items():
            if param not in ["_global_config", "_local_config", "_white_list", "_config_mapping"]:
                result[param] = value
        return result

    @staticmethod
    def get_model_info(model) -> list:
        """Get (node_name, optype) pairs of the model."""
        if not isinstance(model, onnx.ModelProto):
            model = onnx.load(model, load_external_data=False)

        ops = []
        for node in model.graph.node:
            pair = (node.name, node.op_type)
            ops.append(pair)
        logger.debug(f"Get model info: {ops}")
        return ops

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"No such attribute: {key}")

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict):
        """Construct config from a dict.

        Args:
            config_dict: _description_

        Returns:
            The constructed config.
        """
        if constants.GLOBAL not in config_dict and constants.LOCAL not in config_dict:
            config = cls(**config_dict)
            return config
        else:
            config = cls(**config_dict.get(constants.GLOBAL, {}))
            operator_config = config_dict.get(constants.LOCAL, {})
            if operator_config:
                for op_name, op_config in operator_config.items():
                    config.set_local(op_name, cls(**op_config, white_list=constants.EMPTY_WHITE_LIST))
            return config

    def get_diff_dict(self, config) -> Dict[str, Any]:
        """Get the difference between current config and user-specific config."""
        diff_cfg = {}
        for name, cfg in self.get_init_args().items():
            if hasattr(config, name):
                if isinstance(cfg, BaseConfig) and isinstance(config[name], BaseConfig):
                    diff_cfg[name] = cfg.get_diff_dict(config[name])
                elif cfg != config[name]:
                    diff_cfg[name] = cfg
            else:
                diff_cfg[name] = cfg
        return diff_cfg

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
        return cls.from_dict(config_dict)

    def to_json_file(self, filename):
        config_dict = self.to_dict()
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(config_dict, file, indent=4, cls=Encoder)
        logger.info("Dump the config into %s.", filename)

    def to_json_string(self, use_diff: bool = False) -> Union[str, Dict]:
        """Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `BaseConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict(self)
        else:
            config_dict = self.to_dict()
        try:
            return json.dumps(config_dict, indent=2, cls=Encoder) + "\n"
        except Exception as e:
            logger.error("Failed to serialize the config to JSON string: %s", e)
            return config_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    @abstractmethod
    def register_supported_configs(cls):
        """Add all supported configs."""
        raise NotImplementedError

    @classmethod
    def validate(self, user_config: BaseConfig):
        # TODO validate the user config
        pass

    def __add__(self, other: BaseConfig) -> BaseConfig:
        if isinstance(other, type(self)):
            for op_name, config in other.local_config.items():
                self.set_local(op_name, config)
            return self
        else:
            return ComposableConfig(configs=[self, other])

    @staticmethod
    def get_the_default_value_of_param(config: BaseConfig, param: str) -> Any:
        # Get the signature of the __init__ method
        signature = inspect.signature(config.__init__)

        # Get the parameters and their default values
        parameters = signature.parameters
        return parameters.get(param).default if parameters.get(param) is not None else None

    @staticmethod
    def build_tuning_param(config: BaseConfig, param: str):
        # Create `tuning.TuningParam` for each param
        # There are two cases:
        # 1. The param is a string.
        # 2. The param is a `tuning.TuningParam` instance.
        if isinstance(param, str):
            signature = inspect.signature(config.__init__)
            parameters = signature.parameters
            default_param = parameters.get(param).default if parameters.get(param) is not None else None
            tuning_param = TuningParam(name=param, tunable_type=List[type(default_param)])
        elif isinstance(param, TuningParam):
            tuning_param = param
        else:
            raise ValueError(f"Unsupported param type: {param}")
        return tuning_param

    def expand(self) -> List[BaseConfig]:
        """Expand the config.

        Expand rule is:
            1. Expand model_params_list first, then expand params_list
            2. Expand model_params_list/params_list following the order of param order in model_params_list/params_list

            model_params_list=[A, B]                params_list=[C,D]
            A=[1,2], B=[3,4]                        C=[5,6], D=[7,8]

            Expanded results:
                                    --------    Combination 1 (C=5, D=7)
                                   /
                                  / --------    Combination 2 (C=6, D=7)
               Combination 1  ----
                (A=1, B=3)        \ --------    Combination 3 (C=5, D=8)
                                   \
                                    --------    Combination 4 (C=6, D=8)

                                    --------    Combination 1 (C=5, D=7)
                                   /
                                  / --------    Combination 2 (C=6, D=7)
               Combination 2  ----
                (A=2, B=3)        \ --------    Combination 3 (C=5, D=8)
                                   \
                                    --------    Combination 4 (C=6, D=8)

                                    --------    Combination 1 (C=5, D=7)
                                   /
                                  / --------    Combination 2 (C=6, D=7)
               Combination 3  ----
                (A=1, B=4)        \ --------    Combination 3 (C=5, D=8)
                                   \
                                    --------    Combination 4 (C=6, D=8)

                                    --------    Combination 1 (C=5, D=7)
                                   /
                                  / --------    Combination 2 (C=6, D=7)
               Combination 4  ----
                (A=2, B=4)        \ --------    Combination 3 (C=5, D=8)
                                   \
                                    --------    Combination 4 (C=6, D=8)
        """
        config = self
        # set model level params
        model_level_config_lst: List[BaseConfig] = []
        model_params_list = getattr(self, "model_params_list", [])
        tuning_param_list = []
        for param in model_params_list:
            tuning_param = self.build_tuning_param(config, param)
            param_val = getattr(config, tuning_param.name)
            if param_val is not None:
                if tuning_param.is_tunable(param_val):
                    tuning_param.options = param_val
                    tuning_param_list.append(tuning_param)

        if len(tuning_param_list) == 0:
            model_level_config_lst = [config]
        else:
            tuning_param_name_lst = [tuning_param.name for tuning_param in tuning_param_list]
            for params_values in itertools.product(*[tuning_param.options for tuning_param in tuning_param_list[::-1]]):
                new_config = copy.deepcopy(self)
                for param_name, param_value in zip(tuning_param_name_lst[::-1], params_values):
                    setattr(new_config, param_name, param_value)
                logger.debug(new_config.to_dict())
                model_level_config_lst.append(new_config)

        # set op level params
        op_params_list = self.params_list
        op_tuning_param_list = []
        local_op_level_config_lst = []

        for param in op_params_list:
            tuning_param = self.build_tuning_param(config, param)
            param_val = getattr(config, tuning_param.name)
            if param_val is not None:
                if tuning_param.is_tunable(param_val) and len(param_val) > 0:
                    tuning_param.options = param_val
                    op_tuning_param_list.append(tuning_param)

        if len(op_tuning_param_list) == 0:
            local_op_level_config_lst = model_level_config_lst
        else:
            tuning_param_name_lst = [tuning_param.name for tuning_param in op_tuning_param_list]
            tuning_param_val_lst = list(
                itertools.product(*[tuning_param.options for tuning_param in op_tuning_param_list[::-1]])
            )
            tuning_param_pair_lst = [dict(zip(tuning_param_name_lst[::-1], val)) for val in tuning_param_val_lst]

            for model_level_config in model_level_config_lst:
                for tuning_param_pair in tuning_param_pair_lst:
                    new_config = copy.deepcopy(model_level_config)
                    for name, val in tuning_param_pair.items():
                        setattr(new_config, name, val)
                        for _, cfg in new_config.local_config.items():
                            if isinstance(getattr(cfg, name, None), list) and val in getattr(cfg, name, None):
                                setattr(cfg, name, val)
                    logger.debug(new_config.to_dict())
                    local_op_level_config_lst.append(new_config)

        logger.info("Expanded the %s and got %d configs.", self.__class__.name, len(local_op_level_config_lst))
        return local_op_level_config_lst

    def _get_op_name_op_type_config(self):
        op_type_config_dict = dict()
        op_name_config_dict = dict()
        for name, config in self.local_config.items():
            if self._is_op_type(name):
                op_type_config_dict[name] = config
            else:
                op_name_config_dict[name] = config
        return op_type_config_dict, op_name_config_dict

    def to_config_mapping(
        self,
        model: Union[onnx.ModelProto, str],
        config_list: Optional[List[BaseConfig]] = None,
    ) -> OrderedDict[Tuple[str, str], OrderedDict[str, BaseConfig]]:
        if config_list is None:
            config_list = [self]
        model_info = self.get_model_info(model)
        for config in config_list:
            global_config = config.get_params_dict()
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if global_config is not None:
                    self._config_mapping[op_name] = global_config
                if op_type in op_type_config_dict:
                    self._config_mapping[op_name] = op_name_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if isinstance(op_name, str) and re.match(op_name_pattern, op_name):
                        self._config_mapping[op_name] = op_name_config_dict[op_name_pattern]
                    elif op_name_pattern == op_name:
                        self._config_mapping[op_name] = op_name_config_dict[op_name_pattern]
        return self._config_mapping

    @staticmethod
    def _is_op_type(name: str) -> bool:
        return name in constants.STATIC_QOPERATOR_CPU_OP_LIST or name in constants.DYNAMIC_CPU_OP_LIST

    @classmethod
    @abstractmethod
    def get_config_set_for_tuning(cls):
        raise NotImplementedError

    def __eq__(self, other: BaseConfig) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.get_init_args() == other.get_init_args()


class ComposableConfig(BaseConfig):
    name = constants.COMPOSABLE_CONFIG

    def __init__(self, configs: List[BaseConfig]) -> None:
        self.config_list = configs
        self._config_mapping = OrderedDict()

    def __add__(self, other: BaseConfig) -> BaseConfig:
        if isinstance(other, type(self)):
            self.config_list.extend(other.config_list)
        else:
            self.config_list.append(other)
        return self

    def to_dict(self):
        result = {}
        for config in self.config_list:
            result[config.name] = config.to_dict()
        return result

    @classmethod
    def from_dict(cls, config_dict: OrderedDict[str, Dict], config_registry: Dict[str, BaseConfig]):
        assert len(config_dict) >= 1, "The config dict must include at least one configuration."
        num_configs = len(config_dict)
        name, value = next(iter(config_dict.items()))
        config = config_registry[name].from_dict(value)
        for _ in range(num_configs - 1):
            name, value = next(iter(config_dict.items()))
            config += config_registry[name].from_dict(value)
        return config

    def to_json_string(self, use_diff: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=2, cls=Encoder) + "\n"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_config_mapping(
        self, model: Union[onnx.ModelProto, str], config_list: List[BaseConfig] = None
    ) -> OrderedDict[str, BaseConfig]:
        model_info = self.get_model_info(model)
        for config in self.config_list:
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            single_config_model_info = model_info.get(config.name, None)
            for op_name, op_type in single_config_model_info:
                if op_type in op_type_config_dict:
                    self._config_mapping[op_name] = op_name_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        self._config_mapping[op_name] = op_name_config_dict[op_name_pattern]
        return self._config_mapping

    @classmethod
    def register_supported_configs(cls):
        """Add all supported configs."""
        raise NotImplementedError

    @classmethod
    def get_config_set_for_tuning(cls) -> None:
        # TODO handle the composable config in `tuning_config`
        return None

    def get_model_info(self, model, *args, **kwargs):
        model_info_dict = dict()
        for config in self.config_list:
            model_info_dict.update({config.name: config.get_model_info(model, *args, **kwargs)})
        return model_info_dict


def get_all_config_set_from_config_registry() -> List[BaseConfig]:
    all_registered_config_cls: List[Type[BaseConfig]] = config_registry.get_all_config_cls()
    config_set = []
    for config_cls in all_registered_config_cls:
        config_set.append(config_cls.get_config_set_for_tuning())
    return config_set


def register_supported_configs():
    """Register supported configs."""
    all_registered_config_cls: List[Type[BaseConfig]] = config_registry.get_all_config_cls()
    for config_cls in all_registered_config_cls:
        config_cls.register_supported_configs()


@dataclasses.dataclass
class OperatorConfig:
    weight_type: quantization.QuantType
    activation_type: quantization.QuantType
    per_channel: bool
    weight_sym: bool
    activation_sym: bool
    calibrate_method: quantization.CalibrationMethod = quantization.CalibrationMethod.MinMax

    def __post_init__(self):
        self.weight_type = getattr(self.weight_type, "tensor_type", self.weight_type)
        self.activation_type = getattr(self.activation_type, "tensor_type", self.activation_type)
        self.calibrate_method = getattr(self.calibrate_method, "name", self.calibrate_method)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def update(self, kwargs):
        self.weight_type = kwargs.get("weight_type", self.weight_type)
        self.activation_type = kwargs.get("activation_type", self.activation_type)
        self.per_channel = kwargs.get("per_channel", self.per_channel)
        self.weight_sym = kwargs.get("weight_sym", self.weight_sym)
        self.calibrate_method = kwargs.get("calibrate_method", self.calibrate_method)

    def to_dict(self):
        result = {}
        for key, val in self.__dict__.items():
            if not isinstance(val, list):
                result[key] = (
                    getattr(val, "tensor_type", val)
                    if isinstance(val, quantization.QuantType)
                    else getattr(val, "value", val)
                )
            else:
                result[key] = [
                    (
                        getattr(item, "tensor_type", item)
                        if isinstance(item, quantization.QuantType)
                        else getattr(item, "value", item)
                    )
                    for item in val
                ]
        return result

    def __eq__(self, other):
        if isinstance(other, OperatorConfig):
            return self.to_dict() == other.to_dict()
        else:
            return self.to_dict() == other


class _OperatorConfig(NamedTuple):
    config: OperatorConfig
    operators: List[Union[str, Callable]]
    valid_func_list: List[Callable] = []


class BaseWeightOnlyConfig(BaseConfig):
    """Base config class for weight-only quantization."""

    def __init__(
        self,
        weight_dtype: bool = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        providers: List[str] = ["CPUExecutionProvider"],
        quant_last_matmul: bool = True,
        quant_format: quantization.QuantFormat = quantization.QuantFormat.QOperator,
        nodes_to_exclude: list = [],
        white_list: List[Union[str, Callable]] = constants.DEFAULT_WHITE_LIST,
    ):
        """Initialize weight-only quantization config.

        Args:
            weight_dtype (str, optional): Data type for weights, support "uint" and "int", default is "int".
            weight_bits (int, optional): Number of bits used to represent weights, default is 4.
            weight_group_size (int, optional): Size of weight groups, default is 32.
            weight_sym (bool, optional): Indicates whether weights are symmetric, default is True.
            act_dtype (str, optional): Data type for activations, default is "fp32".
            accuracy_level (int, optional): accuracy level. Support 0 (unset), 1(fp32 compute type of jblas kernel),
                2 (fp16 compute type of jblas kernel), 3 (bf16 compute type of jblas kernel),
                4 (int8 compute type of jblas kernel). Defaults to 0.
            ratios (dict, optional): percentile of clip. Defaults to {}.
            providers (list, optional): execution providers to use. Defaults to ["CPUExecutionProvider"].
            quant_last_matmul (bool, optional): whether to quantize the last matmul of the model, default is True.
            quant_format (QuantFormat, optional): use QOperator or QDQ format, default is QOperator.
            nodes_to_exclude (list, optional): nodes in nodes_to_exclude list will be skipped during quantization.
            white_list (list, optional): op in white_list will be applied current config.
                Defaults to constants.DEFAULT_WHITE_LIST.
        """
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.weight_group_size = weight_group_size
        self.weight_sym = weight_sym
        self.act_dtype = act_dtype
        self.accuracy_level = accuracy_level
        self.providers = providers
        self.quant_last_matmul = quant_last_matmul
        self.quant_format = quant_format
        self.nodes_to_exclude = nodes_to_exclude
        super().__init__(white_list=white_list)

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    def to_config_mapping(self, model: Union[onnx.ModelProto, str], config_list: List[BaseConfig] = None):
        if isinstance(model, str):
            model = onnx.load(model, load_external_data=False)

        model_info = self.get_model_info(model)
        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            self._config_mapping.update(config.get_model_params_dict())

            # update node level setting
            last_matmul = None
            global_config = config.get_params_dict()
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            for op_name, op_type in model_info:
                if op_type not in self.white_list:
                    continue

                # skip excluded op
                if any([re.match(exclude_name, op_name) for exclude_name in self.nodes_to_exclude]):
                    continue

                if op_type == "MatMul":
                    last_matmul = op_name

                if global_config is not None:
                    self._config_mapping[op_name] = global_config

                if op_type in op_type_config_dict:
                    self._config_mapping[op_name] = op_type_config_dict[op_type]

                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        self._config_mapping[op_name] = op_name_config_dict[op_name_pattern]

                # convert config to dict
                if op_name in self._config_mapping and hasattr(self._config_mapping[op_name], "to_dict"):
                    self._config_mapping[op_name] = self._config_mapping[op_name].to_dict()

                # update quant_format
                if (
                    ort_version < constants.ONNXRT119_VERSION
                    or model.opset_import[0].version < 21
                    or self._config_mapping[op_name].get("weight_bits", 4) not in [4, 8]
                ):
                    self._config_mapping[op_name].update({"quant_format": quantization.QuantFormat.QOperator})
                if (
                    self._config_mapping[op_name].get("weight_bits", 4) != 4
                    or ort_version < constants.ONNXRT116_VERSION
                    or (
                        ort_version <= constants.ONNXRT1161_VERSION
                        and self._config_mapping[op_name].get("weight_group_size", 32) != 32
                    )
                ):
                    # MatMulFpQ4 support 4 bits and 32 group_size with ort 1.16.0 and 1.16.1 versions
                    # MatMulNBits supports 4 bits and 2^n group_size with ort > 1.16.1
                    del self._config_mapping[op_name]["quant_format"]
        if not self.quant_last_matmul and last_matmul is not None and last_matmul in self._config_mapping:
            del self._config_mapping[last_matmul]
        return self._config_mapping


######################## RNT Config ###############################


@register_config(algo_name=constants.RTN, priority=constants.PRIORITY_RTN)
class RTNConfig(BaseWeightOnlyConfig):
    """Config class for round-to-nearest weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[Union[str, TuningParam]] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
        "ratios",
        "quant_format",
    ]
    model_params_list: List[str] = [
        "providers",
        "layer_wise_quant",
    ]
    name: str = constants.RTN

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
        quant_format: quantization.QuantFormat = quantization.QuantFormat.QOperator,
        nodes_to_exclude: List[str] = [],
        white_list: List[str] = constants.RTN_OP_LIST,
    ):
        """Init RTN weight-only quantization config.

        Args:
            weight_dtype (str, optional): Data type for weights, support "uint" and "int", default is "int".
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
            quant_format (QuantFormat, optional): use QOperator or QDQ format, default is QOperator.
            nodes_to_exclude (list, optional): nodes in nodes_to_exclude list will be skipped during quantization.
            white_list (list, optional): op in white_list will be applied current config.
        """
        self.layer_wise_quant = layer_wise_quant
        self.ratios = ratios

        super().__init__(
            weight_bits=weight_bits,
            weight_dtype=weight_dtype,
            weight_group_size=weight_group_size,
            weight_sym=weight_sym,
            act_dtype=act_dtype,
            accuracy_level=accuracy_level,
            providers=providers,
            quant_last_matmul=quant_last_matmul,
            quant_format=quant_format,
            nodes_to_exclude=nodes_to_exclude,
            white_list=white_list if white_list != constants.RTN_OP_LIST else constants.DEFAULT_WHITE_LIST,
        )
        self.white_list = white_list

    @classmethod
    def register_supported_configs(cls) -> None:
        supported_configs = []
        linear_rtn_config = RTNConfig(
            weight_dtype=["int"],
            weight_bits=[1, 2, 3, 4, 5, 6, 7, 8],
            weight_group_size=[32, -1, 1, 16, 64, 128, 256, 512, 1024],
            weight_sym=[True, False],
            act_dtype=["fp32"],
        )
        operators = constants.RTN_OP_LIST
        supported_configs.append(_OperatorConfig(config=linear_rtn_config, operators=operators))
        cls.supported_configs = supported_configs

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


@register_config(algo_name=constants.GPTQ, priority=constants.PRIORITY_GPTQ)
class GPTQConfig(BaseWeightOnlyConfig):
    """Config class for gptq weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[Union[str, TuningParam]] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
        "quant_format",
    ]
    model_params_list: List[Union[str, TuningParam]] = [
        "percdamp",
        "block_size",
        "actorder",
        "mse",
        "perchannel",
        "providers",
        "layer_wise_quant",
    ]
    name: str = constants.GPTQ

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        weight_group_size: int = 32,
        weight_sym: bool = True,
        act_dtype: str = "fp32",
        accuracy_level: int = 0,
        percdamp: float = 0.01,
        block_size: int = 128,
        actorder: bool = False,
        mse: bool = False,
        perchannel: bool = True,
        providers: List[str] = ["CPUExecutionProvider"],
        layer_wise_quant: bool = False,
        quant_last_matmul: bool = True,
        quant_format: quantization.QuantFormat = quantization.QuantFormat.QOperator,
        nodes_to_exclude: List[str] = [],
        white_list: List[str] = constants.GPTQ_OP_LIST,
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
            block_size (int, optional): execute GPTQ quantization per block. Defaults to 128.
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
            quant_format (QuantFormat, optional): use QOperator or QDQ format, default is QOperator.
            nodes_to_exclude (list, optional): nodes in nodes_to_exclude list will be skipped during quantization.
            white_list (list, optional): op in white_list will be applied current config.
        """
        self.percdamp = percdamp
        self.block_size = block_size
        self.actorder = actorder
        self.mse = mse
        self.perchannel = perchannel
        self.layer_wise_quant = layer_wise_quant

        super().__init__(
            weight_bits=weight_bits,
            weight_dtype=weight_dtype,
            weight_group_size=weight_group_size,
            weight_sym=weight_sym,
            act_dtype=act_dtype,
            accuracy_level=accuracy_level,
            providers=providers,
            quant_last_matmul=quant_last_matmul,
            quant_format=quant_format,
            nodes_to_exclude=nodes_to_exclude,
            white_list=white_list if white_list != constants.GPTQ_OP_LIST else constants.DEFAULT_WHITE_LIST,
        )
        self.white_list = white_list

    @classmethod
    def register_supported_configs(cls) -> None:
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
        operators = constants.GPTQ_OP_LIST
        supported_configs.append(_OperatorConfig(config=linear_gptq_config, operators=operators))
        cls.supported_configs = supported_configs

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


@register_config(algo_name=constants.AWQ, priority=constants.PRIORITY_AWQ)
class AWQConfig(BaseWeightOnlyConfig):
    """Config class for awq weight-only quantization."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        "weight_dtype",
        "weight_bits",
        "weight_group_size",
        "weight_sym",
        "act_dtype",
        "accuracy_level",
        "quant_format",
    ]
    model_params_list: List[str] = [
        "enable_auto_scale",
        "enable_mse_search",
        "providers",
    ]
    name: str = constants.AWQ

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
        quant_format: quantization.QuantFormat = quantization.QuantFormat.QOperator,
        nodes_to_exclude: List[str] = [],
        white_list: List[str] = constants.AWQ_OP_LIST,
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
            quant_format (QuantFormat, optional): use QOperator or QDQ format, default is QOperator.
            nodes_to_exclude (list, optional): nodes in nodes_to_exclude list will be skipped during quantization.
            white_list (list, optional): op in white_list will be applied current config.
        """
        self.enable_auto_scale = enable_auto_scale
        self.enable_mse_search = enable_mse_search

        super().__init__(
            weight_bits=weight_bits,
            weight_dtype=weight_dtype,
            weight_group_size=weight_group_size,
            weight_sym=weight_sym,
            act_dtype=act_dtype,
            accuracy_level=accuracy_level,
            providers=providers,
            quant_last_matmul=quant_last_matmul,
            quant_format=quant_format,
            nodes_to_exclude=nodes_to_exclude,
            white_list=white_list if white_list != constants.AWQ_OP_LIST else constants.DEFAULT_WHITE_LIST,
        )
        self.white_list = white_list

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
        operators = constants.AWQ_OP_LIST
        supported_configs.append(_OperatorConfig(config=linear_awq_config, operators=operators))
        cls.supported_configs = supported_configs

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


######################## WOQ Tuning Config ###############################


def get_woq_tuning_config(quant_format=quantization.QuantFormat.QOperator) -> list:
    """Generate the config set for WOQ tuning.

    Returns:
        the list of WOQ quant config.
    """
    RTN_G32ASYM = RTNConfig(weight_sym=False, quant_format=quant_format)
    GPTQ_G32ASYM = GPTQConfig(weight_sym=False, quant_format=quant_format)
    GPTQ_G32ASYM_DISABLE_LAST_MATMUL = GPTQConfig(weight_sym=False, quant_last_matmul=False, quant_format=quant_format)
    GPTQ_G128ASYM = GPTQConfig(weight_group_size=128, weight_sym=False, quant_format=quant_format)
    AWQ_G32ASYM = AWQConfig(weight_sym=False, quant_format=quant_format)
    return [RTN_G32ASYM, GPTQ_G32ASYM, GPTQ_G32ASYM_DISABLE_LAST_MATMUL, GPTQ_G128ASYM, AWQ_G32ASYM]


##################### Config for ONNXRuntime-like user-facing API ############


class ExtraOptions:
    def __init__(
        self,
        ActivationSymmetric=False,
        WeightSymmetric=True,
        AddQDQPairToWeight=False,
        OpTypesToExcludeOutputQuantization=[],
        DedicatedQDQPair=False,
        SmoothQuant=False,
        SmoothQuantAlpha=0.5,
        SmoothQuantFolding=True,
        SmoothQuantOpTypes=["Gemm", "Conv", "MatMul", "FusedConv"],
        SmoothQuantCalibIter=100,
        SmoothQuantScalesPerOp=True,
        **kwargs,
    ):
        self.ActivationSymmetric = ActivationSymmetric
        self.WeightSymmetric = WeightSymmetric
        self.AddQDQPairToWeight = AddQDQPairToWeight
        self.OpTypesToExcludeOutputQuantization = OpTypesToExcludeOutputQuantization
        self.DedicatedQDQPair = DedicatedQDQPair
        self.SmoothQuant = SmoothQuant
        self.SmoothQuantAlpha = SmoothQuantAlpha
        self.SmoothQuantFolding = SmoothQuantFolding
        self.SmoothQuantOpTypes = SmoothQuantOpTypes
        self.SmoothQuantCalibIter = SmoothQuantCalibIter
        self.SmoothQuantScalesPerOp = SmoothQuantScalesPerOp


def static_basic_check(config, optype, execution_provider, quant_format):
    if getattr(quant_format, "value", quant_format) == 0:
        if execution_provider not in constants.STATIC_QOPERATOR_OP_LIST_MAP:
            raise ValueError(
                "Unsupported execution_provider {}, only support {}.".format(
                    execution_provider, list(constants.STATIC_QOPERATOR_OP_LIST_MAP.keys())
                )
            )
        supported_optype = constants.STATIC_QOPERATOR_OP_LIST_MAP[execution_provider]
        if optype not in supported_optype:
            raise ValueError(
                "Unsupported optype {} for {}, only support {}.".format(optype, execution_provider, supported_optype)
            )
    elif getattr(quant_format, "value", quant_format) == 1:
        if execution_provider not in constants.STATIC_QDQ_OP_LIST_MAP:
            raise ValueError(
                "Unsupported execution_provider {}, only support {}.".format(
                    execution_provider, list(constants.STATIC_QDQ_OP_LIST_MAP.keys())
                )
            )
        supported_optype = constants.STATIC_QDQ_OP_LIST_MAP[execution_provider]
        if optype not in supported_optype:
            raise ValueError(
                "Unsupported optype {} for {}, only support {}.".format(optype, execution_provider, supported_optype)
            )
    else:
        raise ValueError(
            "Unsupported quant_format {}, only support QuantFormat.QOperator and QuantFormat.QDQ.".format(quant_format)
        )
    return config


def static_cpu_check(config, optype, execution_provider, quant_format):
    if execution_provider != "CPUExecutionProvider":
        return config

    # only support per-tensor
    if optype in [
        "EmbedLayerNormalization",
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
        "Tile",
        "Unsqueeze",
        "Transpose",
        "Resize",
        "Abs",
        "Shrink",
        "Sign",
        "Attention",
        "Flatten",
        "Expand",
        "Slice",
        "Mod",
        "ReduceMax",
        "ReduceMin",
        "CenterCropPad",
        "Add",
        "Mul",
        "ArgMax",
    ]:
        setattr(config, "per_channel", False)

    if optype in ["Attention"]:
        setattr(config, "activation_type", onnx.TensorProto.UINT8)
    return config


def static_cuda_check(config, optype, execution_provider, quant_format):
    if execution_provider != "CUDAExecutionProvider":
        return config

    # only support per-tensor
    if optype in [
        "EmbedLayerNormalization",
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
        "Tile",
        "Unsqueeze",
        "Transpose",
        "Resize",
        "Abs",
        "Shrink",
        "Sign",
        "Attention",
        "Flatten",
        "Expand",
        "Slice",
        "Mod",
        "ReduceMax",
        "ReduceMin",
        "CenterCropPad",
        "Add",
        "Mul",
        "ArgMax",
    ]:
        setattr(config, "per_channel", False)

    if optype in ["Attention"]:
        setattr(config, "activation_type", onnx.TensorProto.INT8)
        setattr(config, "weight_type", onnx.TensorProto.INT8)
    return config


def static_dml_check(config, optype, execution_provider, quant_format):
    if execution_provider != "DmlExecutionProvider":
        return config

    # only support per-tensor
    if optype in ["Conv", "MatMul", "Mul", "Relu", "Clip", "MaxPool", "Add"]:
        setattr(config, "per_channel", False)
    return config


def static_dnnl_check(config, optype, execution_provider, quant_format):
    if execution_provider != "DnnlExecutionProvider":
        return config

    # current configurations are same as CPU EP
    return static_cpu_check(config, optype, execution_provider, quant_format)


def static_trt_check(config, optype, execution_provider, quant_format):
    if execution_provider != "TensorrtExecutionProvider":
        return config

    # only support S8S8
    if optype in ["Conv", "MatMul", "Gather", "Gemm"]:
        setattr(config, "weight_type", onnx.TensorProto.INT8)
        setattr(config, "weight_sym", True)
        setattr(config, "activation_type", onnx.TensorProto.INT8)
        setattr(config, "activation_sym", True)
        setattr(config, "per_channel", [False, True])
    else:
        setattr(config, "weight_type", onnx.TensorProto.INT8)
        setattr(config, "weight_sym", True)
        setattr(config, "activation_type", onnx.TensorProto.INT8)
        setattr(config, "activation_sym", True)
    return config


STATIC_CHECK_FUNC_LIST = [
    static_basic_check,
    static_cpu_check,
    static_cuda_check,
    static_dml_check,
    static_dnnl_check,
    static_trt_check,
]


def dynamic_basic_check(config, optype, execution_provider, quant_format=None):
    if execution_provider not in constants.DYNAMIC_OP_LIST_MAP:
        raise ValueError(
            "Unsupported execution_provider {}, only support {}.".format(
                execution_provider, list(constants.DYNAMIC_OP_LIST_MAP.keys())
            )
        )

    supported_optype = constants.DYNAMIC_OP_LIST_MAP[execution_provider]
    if optype not in supported_optype:
        raise ValueError(
            "Unsupported optype {} for {}, only support {}.".format(optype, execution_provider, supported_optype)
        )
    return config


def dynamic_cpu_check(config, optype, execution_provider, quant_format=None):
    if execution_provider != "CPUExecutionProvider":
        return config
    # TODO: add constraints for other EP
    if optype in ["FusedConv", "Conv", "EmbedLayerNormalization", "Gather", "Attention", "LSTM"]:
        setattr(config, "per_channel", False)
    return config


def dynamic_cuda_check(config, optype, execution_provider, quant_format=None):
    if execution_provider != "CUDAExecutionProvider":
        return config
    # current configurations are same as CPU EP
    return dynamic_cpu_check(config, optype, execution_provider, quant_format)


def dynamic_dml_check(config, optype, execution_provider, quant_format=None):
    if execution_provider != "DmlExecutionProvider":
        return config

    # don't support dynamic quantization
    return None


def dynamic_dnnl_check(config, optype, execution_provider, quant_format=None):
    if execution_provider != "DnnlExecutionProvider":
        return config
    # current configurations are same as CPU EP
    return dynamic_cpu_check(config, optype, execution_provider, quant_format)


def dynamic_trt_check(config, optype, execution_provider, quant_format=None):
    if execution_provider != "TensorrtExecutionProvider":
        return config

    # don't support dynamic quantization
    return None


DYNAMIC_CHECK_FUNC_LIST = [
    dynamic_basic_check,
    dynamic_cpu_check,
    dynamic_cuda_check,
    dynamic_dml_check,
    dynamic_dnnl_check,
    dynamic_trt_check,
]


@register_config(algo_name=constants.STATIC_QUANT, priority=constants.PRIORITY_STATIC_QUANT)
class StaticQuantConfig(BaseConfig, ort_quant.StaticQuantConfig):

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        "weight_type",
        "activation_type",
        "per_channel",
        "weight_sym",
        "activation_sym",
        "calibrate_method",
    ]
    model_params_list: List[str] = [
        "quant_format",
        "reduce_range",
        "use_external_data_format",
        "calibration_sampling_size",
        "quant_last_matmul",
    ]
    name: str = constants.STATIC_QUANT

    def __init__(
        self,
        calibration_data_reader: data_reader.CalibrationDataReader = None,
        calibrate_method=quantization.CalibrationMethod.MinMax,
        quant_format=quantization.QuantFormat.QOperator,
        activation_type=quantization.QuantType.QUInt8,
        weight_type=quantization.QuantType.QInt8,
        op_types_to_quantize=None,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        per_channel=False,
        reduce_range=False,
        use_external_data_format=False,
        extra_options=None,
        calibration_sampling_size=100,
        quant_last_matmul=True,
        execution_provider=None,
        **kwargs,
    ):
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
        if execution_provider is None:
            execution_provider = utility.auto_detect_ep()
        if op_types_to_quantize is None:
            op_types_to_quantize = (
                constants.STATIC_QOPERATOR_OP_LIST_MAP.get(execution_provider, [])
                if quant_format == quantization.QuantFormat.QOperator
                else constants.STATIC_QDQ_OP_LIST_MAP.get(execution_provider, [])
            )
        if not reduce_range and not utility.CpuInfo().vnni:
            logger.warning(
                "VNNI is not supported and reduce_range=False, reduce_range=True is recommended to avoid potential accuracy issue."
            )
        ort_quant.StaticQuantConfig.__init__(
            self,
            calibration_data_reader=calibration_data_reader,
            calibrate_method=calibrate_method,
            quant_format=quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            per_channel=per_channel,
            reduce_range=reduce_range,
            use_external_data_format=use_external_data_format,
            extra_options=extra_options,
        )
        # do not load TensorRT if backend is not TensorrtExecutionProvider
        if "TensorrtExecutionProvider" in execution_provider:
            logger.info("Update some parameters for TensorrtExecutionProvider")
            os.environ["ORT_TENSORRT_INT8_ENABLE"] = "0"
            self.extra_options.update(
                {
                    "AddQDQPairToWeight": True,
                    "DedicatedQDQPair": True,
                    "OpTypesToExcludeOutputQuantization": ["Conv", "Gemm", "Add", "MatMul"],
                }
            )
        else:
            os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"

        self.execution_provider = execution_provider
        self.quant_last_matmul = quant_last_matmul
        self.calibration_sampling_size = calibration_sampling_size
        _extra_options = ExtraOptions(**self.extra_options)
        self.weight_sym = _extra_options.WeightSymmetric
        self.activation_sym = _extra_options.ActivationSymmetric
        self.optypes_to_exclude_output_quant = _extra_options.OpTypesToExcludeOutputQuantization
        self.dedicated_qdq_pair = _extra_options.DedicatedQDQPair
        self.add_qdq_pair_to_weight = _extra_options.AddQDQPairToWeight
        BaseConfig.__init__(self, white_list=self.op_types_to_quantize)

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    def _post_init(self):
        for op_name_or_type in self.op_types_to_quantize:
            params = self.get_params_dict()
            op_config = OperatorConfig(**params)

            for valid_func in STATIC_CHECK_FUNC_LIST:
                op_config = valid_func(op_config, op_name_or_type, self.execution_provider, self.quant_format)
            self.set_local(op_name_or_type, op_config)

    def to_config_mapping(self, model: Union[onnx.ModelProto, str], config_list: List[BaseConfig] = None):
        if isinstance(model, str):
            model = onnx.load(model, load_external_data=False)

        model_info = self.get_model_info(model)

        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            self._config_mapping.update(config.get_model_params_dict())

            # update node level setting
            global_config = config.global_config
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            last_matmul = None
            for op_name, op_type in model_info:
                if op_type == "MatMul":
                    last_matmul = op_name
                if (
                    isinstance(self.op_types_to_quantize, list)
                    and len(self.op_types_to_quantize) > 0
                    and op_type not in self.op_types_to_quantize
                ):
                    continue
                if (
                    isinstance(self.nodes_to_quantize, list)
                    and len(self.nodes_to_quantize) > 0
                    and op_name not in self.nodes_to_quantize
                ):
                    continue
                if (
                    isinstance(self.nodes_to_exclude, list)
                    and len(self.nodes_to_exclude) > 0
                    and op_name in self.nodes_to_exclude
                ):
                    continue
                if op_type in op_type_config_dict:
                    self._config_mapping[op_name] = op_type_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        self._config_mapping[op_name] = op_name_config_dict[op_name_pattern]

        if not self.quant_last_matmul and last_matmul is not None and last_matmul in self._config_mapping:
            del self._config_mapping[last_matmul]
        return self._config_mapping

    @classmethod
    def get_config_set_for_tuning(
        cls,
        quant_format=quantization.QuantFormat.QOperator,
        activation_type=quantization.QuantType.QUInt8,
        weight_type=quantization.QuantType.QInt8,
        execution_provider=None,
        op_types_to_quantize=None,
        nodes_to_exclude=None,
        reduce_range=False,
        use_external_data_format=False,
        calibration_sampling_size=100,
        quant_last_matmul=True,
        **kwargs,
    ) -> Union[None, "StaticQuantConfig", List["StaticQuantConfig"]]:  # pragma: no cover
        if execution_provider is None:
            execution_provider = utility.auto_detect_ep()
        StaticQuantConfig.register_supported_configs()
        if op_types_to_quantize is None:
            op_types_to_quantize = (
                constants.STATIC_QOPERATOR_OP_LIST_MAP.get(execution_provider, [])
                if quant_format == quantization.QuantFormat.QOperator
                else constants.STATIC_QDQ_OP_LIST_MAP.get(execution_provider, [])
            )

        op_type_candidate = [
            op_types_to_quantize,
            list(set(op_types_to_quantize).difference({"Add", "Mul"})),
            list(set(op_types_to_quantize).difference({"Add", "Mul", "Gather", "GatherElements", "GatherND"})),
            list(
                set(op_types_to_quantize).difference(
                    {"Add", "Mul", "Gather", "GatherElements", "GatherND", "Attention"}
                )
            ),
        ]

        cfg_lst = []
        for item in op_type_candidate:
            cfg_lst.append(
                StaticQuantConfig(
                    activation_type=activation_type,
                    weight_type=weight_type,
                    execution_provider=execution_provider,
                    quant_format=quant_format,
                    reduce_range=reduce_range,
                    use_external_data_format=use_external_data_format,
                    calibration_sampling_size=calibration_sampling_size,
                    op_types_to_quantize=item,
                    nodes_to_exclude=nodes_to_exclude,
                    quant_last_matmul=[True, False],
                    per_channel=[True, False],
                    **kwargs,
                )
            )
        return cfg_lst

    @classmethod
    def register_supported_configs(cls) -> None:
        supported_configs = []
        supported_configs.append(
            _OperatorConfig(
                config=OperatorConfig(
                    weight_type=onnx.TensorProto.UINT8,
                    weight_sym=False,
                    per_channel=[True, False],
                    calibrate_method=[
                        quantization.CalibrationMethod.MinMax,
                        quantization.CalibrationMethod.Entropy,
                        quantization.CalibrationMethod.Percentile,
                    ],
                    activation_type=onnx.TensorProto.UINT8,
                    activation_sym=False,
                ),
                operators=["GatherND", "GatherElements", "Gather"],
                valid_func_list=STATIC_CHECK_FUNC_LIST,
            )
        )
        supported_configs.append(
            _OperatorConfig(
                config=OperatorConfig(
                    weight_type=onnx.TensorProto.UINT8,
                    weight_sym=False,
                    per_channel=False,
                    calibrate_method=[
                        quantization.CalibrationMethod.MinMax,
                        quantization.CalibrationMethod.Entropy,
                        quantization.CalibrationMethod.Percentile,
                    ],
                    activation_type=onnx.TensorProto.UINT8,
                    activation_sym=False,
                ),
                operators=["EmbedLayerNormalization"],
                valid_func_list=STATIC_CHECK_FUNC_LIST,
            )
        )
        supported_configs.append(
            _OperatorConfig(
                config=OperatorConfig(
                    weight_type=onnx.TensorProto.INT8,
                    weight_sym=True,
                    per_channel=[True, False],
                    calibrate_method=[
                        quantization.CalibrationMethod.MinMax,
                        quantization.CalibrationMethod.Entropy,
                        quantization.CalibrationMethod.Percentile,
                    ],
                    activation_type=onnx.TensorProto.UINT8,
                    activation_sym=False,
                ),
                operators=["Conv", "MatMul", "Gemm", "FusedConv"],
                valid_func_list=STATIC_CHECK_FUNC_LIST,
            )
        )
        supported_configs.append(
            _OperatorConfig(
                config=OperatorConfig(
                    weight_type=onnx.TensorProto.INT8,
                    weight_sym=True,
                    per_channel=False,
                    calibrate_method=[
                        quantization.CalibrationMethod.MinMax,
                        quantization.CalibrationMethod.Entropy,
                        quantization.CalibrationMethod.Percentile,
                    ],
                    activation_type=onnx.TensorProto.UINT8,
                    activation_sym=False,
                ),
                operators=[
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
                    "Tile",
                    "Unsqueeze",
                    "Transpose",
                    "Resize",
                    "Abs",
                    "Shrink",
                    "Sign",
                    "Attention",
                    "Flatten",
                    "Expand",
                    "Slice",
                    "Mod",
                    "ReduceMax",
                    "ReduceMin",
                    "CenterCropPad",
                    "Add",
                    "Mul",
                    "ArgMax",
                ],
                valid_func_list=STATIC_CHECK_FUNC_LIST,
            )
        )
        cls.supported_configs = supported_configs


######################## SmoohQuant Config ###############################


@register_config(algo_name=constants.SMOOTH_QUANT, priority=constants.PRIORITY_SMOOTH_QUANT)
class SmoothQuantConfig(StaticQuantConfig):
    """Smooth quant quantization config."""

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        "weight_type",
        "activation_type",
        "per_channel",
        "weight_sym",
        "activation_sym",
        "calibrate_method",
    ]
    model_params_list: List[str] = [
        # smooth parameters
        "alpha",
        "folding",
        "auto_alpha_args",
        "calib_iter",
        "scales_per_op",
    ]
    name: str = constants.SMOOTH_QUANT

    def __init__(
        self,
        alpha: float = 0.5,
        folding: bool = True,
        op_types: List[str] = ["Gemm", "Conv", "MatMul", "FusedConv"],
        calib_iter: int = 100,
        scales_per_op: bool = True,
        auto_alpha_args: dict = {"alpha_min": 0.3, "alpha_max": 0.7, "alpha_step": 0.05, "attn_method": "min"},
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
            kwargs (dict): kwargs in below link are supported except calibration_data_reader:
                https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py#L78
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.folding = folding
        self.op_types = op_types
        self.calib_iter = calib_iter
        self.scales_per_op = scales_per_op
        self.auto_alpha_args = auto_alpha_args

    @classmethod
    def register_supported_configs(cls) -> List[_OperatorConfig]:
        supported_configs = []
        smooth_quant_config = SmoothQuantConfig()
        operators = ["Gemm", "Conv", "MatMul", "FusedConv"]
        supported_configs.append(_OperatorConfig(config=smooth_quant_config, operators=operators))
        cls.supported_configs = supported_configs

    @classmethod
    def get_config_set_for_tuning(
        cls,
    ) -> Union[None, "SmoothQuantConfig", List["SmoothQuantConfig"]]:  # pragma: no cover
        return SmoothQuantConfig(alpha=np.arange(0.3, 0.7, 0.05))


def get_default_sq_config() -> SmoothQuantConfig:
    """Generate the default smooth quant config.

    Returns:
        the default smooth quant config.
    """
    return SmoothQuantConfig()


@register_config(algo_name=constants.DYNAMIC_QUANT, priority=constants.PRIORITY_DYNAMIC_QUANT)
class DynamicQuantConfig(BaseConfig, ort_quant.DynamicQuantConfig):
    """This is a class for dynamic Quant Configuration.

    Inherit from DynamicQuantConfig:
        https://github.com/microsoft/onnxruntime/blob/v1.17.1/onnxruntime/python/tools/quantization/quantize.py#L206
    """

    supported_configs: List[_OperatorConfig] = []
    params_list: List[str] = [
        "weight_type",
        "activation_type",
        "per_channel",
        "weight_sym",
        "activation_sym",
    ]
    model_params_list: List[str] = [
        "reduce_range",
        "use_external_data_format",
        "quant_last_matmul",
    ]
    name: str = constants.DYNAMIC_QUANT

    def __init__(
        self,
        weight_type: quantization.QuantType = quantization.QuantType.QInt8,
        op_types_to_quantize: List[str] = None,
        nodes_to_quantize: List[str] = None,
        nodes_to_exclude: List[str] = None,
        per_channel: bool = False,
        reduce_range: bool = False,
        use_external_data_format: bool = False,
        extra_options: dict = None,
        quant_last_matmul: bool = True,
        execution_provider: str = None,
        **kwargs,
    ):
        if execution_provider is None:
            execution_provider = utility.auto_detect_ep()
        if op_types_to_quantize is None:
            op_types_to_quantize = constants.DYNAMIC_OP_LIST_MAP.get(execution_provider, [])
        if not reduce_range and not utility.CpuInfo().vnni:
            logger.warning(
                "VNNI is not supported and reduce_range=False, reduce_range=True is recommended to avoid potential accuracy issue."
            )
        ort_quant.DynamicQuantConfig.__init__(
            self,
            weight_type=weight_type,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            per_channel=per_channel,
            reduce_range=reduce_range,
            use_external_data_format=use_external_data_format,
            extra_options=extra_options,
        )
        self.execution_provider = execution_provider
        self.quant_last_matmul = quant_last_matmul
        self.activation_type = quantization.QuantType.QUInt8
        _extra_options = ExtraOptions(**self.extra_options)
        self.weight_sym = _extra_options.WeightSymmetric
        self.activation_sym = _extra_options.ActivationSymmetric
        BaseConfig.__init__(self, white_list=op_types_to_quantize)

    def get_model_params_dict(self):
        result = dict()
        for param in self.model_params_list:
            result[param] = getattr(self, param)
        return result

    def _post_init(self):
        for op_name_or_type in self.op_types_to_quantize:
            params = self.get_params_dict()
            op_config = OperatorConfig(**params)
            for valid_func in DYNAMIC_CHECK_FUNC_LIST:
                op_config = valid_func(op_config, op_name_or_type, self.execution_provider)
            self.set_local(op_name_or_type, op_config)

    def to_config_mapping(self, model: Union[onnx.ModelProto, str], config_list: List[BaseConfig] = None):
        if isinstance(model, str):
            model = onnx.load(model, load_external_data=False)

        model_info = self.get_model_info(model)

        if config_list is None:
            config_list = [self]
        for config in config_list:
            # update model level setting
            self._config_mapping.update(config.get_model_params_dict())

            # update node level setting
            op_type_config_dict, op_name_config_dict = config._get_op_name_op_type_config()
            last_matmul = None
            for op_name, op_type in model_info:
                if op_type == "MatMul":
                    last_matmul = op_name
                if (
                    isinstance(self.op_types_to_quantize, list)
                    and len(self.op_types_to_quantize) > 0
                    and op_type not in self.op_types_to_quantize
                ):
                    continue
                if (
                    isinstance(self.nodes_to_quantize, list)
                    and len(self.nodes_to_quantize) > 0
                    and op_name not in self.nodes_to_quantize
                ):
                    continue
                if (
                    isinstance(self.nodes_to_exclude, list)
                    and len(self.nodes_to_exclude) > 0
                    and op_name in self.nodes_to_exclude
                ):
                    continue
                if op_type in op_type_config_dict:
                    self._config_mapping[op_name] = op_type_config_dict[op_type]
                for op_name_pattern in op_name_config_dict:
                    if re.match(op_name_pattern, op_name):
                        self._config_mapping[op_name] = op_name_config_dict[op_name_pattern]

        if not self.quant_last_matmul and last_matmul is not None and last_matmul in self._config_mapping:
            del self._config_mapping[last_matmul]
        return self._config_mapping

    @classmethod
    def get_config_set_for_tuning(
        cls,
        weight_type=quantization.QuantType.QInt8,
        execution_provider=None,
        op_types_to_quantize: List[str] = None,
        nodes_to_exclude: List[str] = None,
        reduce_range: bool = False,
        use_external_data_format: bool = False,
        quant_last_matmul: bool = True,
    ) -> Union[None, "DynamicQuantConfig", List["DynamicQuantConfig"]]:  # pragma: no cover
        if execution_provider is None:
            execution_provider = utility.auto_detect_ep()
        if op_types_to_quantize is None:
            op_types_to_quantize = constants.DYNAMIC_OP_LIST_MAP.get(execution_provider, [])

        op_type_candidate = [
            op_types_to_quantize,
            list(set(op_types_to_quantize).difference({"EmbedLayerNormalization", "Gather", "LSTM"})),
            list(
                set(op_types_to_quantize).difference({"EmbedLayerNormalization", "Gather", "LSTM", "Conv", "FusedConv"})
            ),
            list(
                set(op_types_to_quantize).difference(
                    {"EmbedLayerNormalization", "Gather", "LSTM", "Conv", "FusedConv", "Attention"}
                )
            ),
            list(
                set(op_types_to_quantize).difference(
                    {"EmbedLayerNormalization", "Gather", "LSTM", "Conv", "FusedConv", "MatMul"}
                )
            ),
        ]

        cfg_lst = []
        for item in op_type_candidate:
            cfg_lst.append(
                DynamicQuantConfig(
                    weight_type=weight_type,
                    execution_provider=execution_provider,
                    op_types_to_quantize=item,
                    nodes_to_exclude=nodes_to_exclude,
                    reduce_range=reduce_range,
                    use_external_data_format=use_external_data_format,
                    quant_last_matmul=[True, False],
                    per_channel=[True, False],
                )
            )
        return cfg_lst

    @classmethod
    def register_supported_configs(cls) -> None:
        supported_configs = []
        supported_configs.append(
            _OperatorConfig(
                config=OperatorConfig(
                    weight_type=onnx.TensorProto.UINT8,
                    weight_sym=False,
                    per_channel=False,
                    activation_type=onnx.TensorProto.UINT8,
                    activation_sym=False,
                ),
                operators=["FusedConv", "Conv", "EmbedLayerNormalization"],
                valid_func_list=DYNAMIC_CHECK_FUNC_LIST,
            )
        )
        supported_configs.append(
            _OperatorConfig(
                config=OperatorConfig(
                    weight_type=onnx.TensorProto.INT8,
                    weight_sym=True,
                    per_channel=[True, False],
                    activation_type=onnx.TensorProto.UINT8,
                    activation_sym=False,
                ),
                operators=["MatMul"],
                valid_func_list=DYNAMIC_CHECK_FUNC_LIST,
            )
        )
        supported_configs.append(
            _OperatorConfig(
                config=OperatorConfig(
                    weight_type=onnx.TensorProto.INT8,
                    weight_sym=True,
                    per_channel=False,
                    activation_type=onnx.TensorProto.UINT8,
                    activation_sym=False,
                ),
                operators=["Gather", "Attention", "LSTM"],
                valid_func_list=DYNAMIC_CHECK_FUNC_LIST,
            )
        )
        cls.supported_configs = supported_configs


##################### NC Algo Configs End ###################################

register_supported_configs()
