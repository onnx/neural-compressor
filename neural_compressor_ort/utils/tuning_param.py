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

import typing
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel

from neural_compressor_ort.utils import logger


class ParamLevel(Enum):
    OP_LEVEL = auto()
    OP_TYPE_LEVEL = auto()
    MODEL_LEVEL = auto()


class TuningParam:
    """Define the tunable parameter for the algorithm.

    Example:
        Class FakeAlgoConfig(BaseConfig):
            '''Fake algo config.'''.

            params_list = [
                ...
                # For simple tunable types, like a list of int, giving
                # the param name is enough. `BaseConfig` class will
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

        class DynamicInputArgsModel(BaseModel):
            input_args: expect_args_type

        return DynamicInputArgsModel

    def is_tunable(self, value: Any) -> bool:
        # Use `Pydantic` to validate the input_args.
        # TODO: refine the implementation in further.
        assert isinstance(
            self.tunable_type, typing._GenericAlias
        ), f"Expected a type hint, got {self.tunable_type} instead."
        DynamicInputArgsModel = TuningParam.create_input_args_model(self.tunable_type)
        try:
            new_args = DynamicInputArgsModel(input_args=value)
            return True
        except Exception as e:
            logger.debug(f"Failed to validate the input_args: {e}")
            return False
