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

import abc


class CalibrationDataReader(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "get_next") and callable(subclass.get_next) or NotImplemented

    @abc.abstractmethod
    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        result = self.get_next()
        if result is None:
            raise StopIteration
        return result

    @abc.abstractmethod
    def rewind(self):
        """Regenerate data."""
        raise NotImplementedError
