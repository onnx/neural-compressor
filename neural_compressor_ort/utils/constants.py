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

# All constants

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
SMOOTH_QUANT = "smooth_quant"
GPTQ = "gptq"
AWQ = "awq"  # pragma: no cover

# options
import datetime

DEFAULT_WORKSPACE = "./nc_workspace/{}/".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

from typing import Callable, Union

OP_NAME_OR_MODULE_TYPE = Union[str, Callable]

from packaging.version import Version

ONNXRT116_VERSION = Version("1.16.0")
ONNXRT1161_VERSION = Version("1.16.1")

PRIORITY_RTN = 60
PRIORITY_GPTQ = 70
PRIORITY_AWQ = 50
PRIORITY_SMOOTH_QUANT = 80

MAXIMUM_PROTOBUF = 2147483648

WHITE_MODULE_LIST = ["MatMul", "Conv"]
