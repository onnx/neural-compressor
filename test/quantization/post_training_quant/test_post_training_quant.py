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

import functools
import glob
import os
import shutil
import unittest
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
from optimum.exporters.onnx import main_export

from onnx_neural_compressor import data_reader, quantization
from onnx_neural_compressor.quantization import config

from typing import Callable, Dict, List, Optional, Union  # isort: skip


def fake_eval(model, eval_result_lst):
    acc = eval_result_lst.pop(0)
    return acc


class DataReader(data_reader.CalibrationDataReader):

    def __init__(self, model):
        model = onnx.load(model)
        batch_size = 1
        sequence_length = 1
        self.data = {
            "input_ids": np.random.randint(10, size=(batch_size, sequence_length)).astype("int64"),
            "attention_mask": np.zeros((batch_size, sequence_length)).astype("int64"),
        }
        for inp in model.graph.input:
            if inp.name in self.data:
                continue
            if inp.name == "position_ids":
                # model is exported with optimum >= 1.14.0 with new input 'position_ids'
                self.data[inp.name] = np.random.randint(10, size=(batch_size, sequence_length)).astype("int64")

        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([self.data])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def _count_op_num(model, optype):
    num = 0
    for node in model.graph.node:
        if node.op_type == optype:
            num += 1
    return num


class TestStaticQuant(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="model",
        )
        self.model = glob.glob(os.path.join("./model", "*.onnx"))[0]
        self.data_reader = DataReader(self.model)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./model", ignore_errors=True)
        os.remove("quant.onnx")

    def test_static_quant(self):
        cfg = config.StaticQuantConfig(
            calibration_data_reader=self.data_reader,
            weight_type=quantization.QuantType.QInt8,
            per_channel=True,
            quant_last_matmul=True,
            extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
            execution_provider="CPUExecutionProvider",
        )
        quantization.quantize(self.model, "quant.onnx", cfg)
        q_model = onnx.load("quant.onnx")
        qmatmul_num_enable_last = _count_op_num(q_model, "QLinearMatMul")

        cfg = config.StaticQuantConfig(
            calibration_data_reader=self.data_reader,
            weight_type=quantization.QuantType.QInt8,
            per_channel=True,
            quant_last_matmul=False,
            extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
            execution_provider="CPUExecutionProvider",
        )
        quantization.quantize(self.model, "quant.onnx", cfg)
        q_model = onnx.load("quant.onnx")
        node_num_basic = len(q_model.graph.node)
        qmatmul_num_disable_last = _count_op_num(q_model, "QLinearMatMul")

        # check quant_last_matmul work
        self.assertEqual(qmatmul_num_enable_last, qmatmul_num_disable_last + 1)

        cfg = config.StaticQuantConfig(
            calibration_data_reader=self.data_reader,
            weight_type=quantization.QuantType.QUInt8,
            per_channel=False,
            quant_last_matmul=False,
            extra_options={"WeightSymmetric": False, "ActivationSymmetric": True},
            execution_provider="CPUExecutionProvider",
        )
        quantization.quantize(self.model, "quant.onnx", cfg, ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
        q_model = onnx.load("quant.onnx")
        node_num_extended = len(q_model.graph.node)


        # check graph optimization work
        self.assertGreater(node_num_basic, node_num_extended)


        # check op_types_to_quantize work
        cfg = config.StaticQuantConfig(
            calibration_data_reader=self.data_reader,
            weight_type=quantization.QuantType.QUInt8,
            per_channel=False,
            quant_last_matmul=False,
            op_types_to_quantize=["MatMul", "Gather"],
            extra_options={"WeightSymmetric": False, "ActivationSymmetric": True},
            execution_provider="CPUExecutionProvider",
        )
        quantization.quantize(self.model, "quant.onnx", cfg)
        q_model = onnx.load("quant.onnx")
        self.assertEqual(_count_op_num(q_model, "QLinearAdd"), 0)
        self.assertGreater(_count_op_num(q_model, "QLinearMatMul"), 0)

        # check nodes_to_quantize work
        quantizable_matmuls = [i.name.split("_quant")[0] for i in q_model.graph.node if i.op_type == "QLinearMatMul"]
        cfg = config.StaticQuantConfig(
            calibration_data_reader=self.data_reader,
            weight_type=quantization.QuantType.QUInt8,
            nodes_to_quantize=[quantizable_matmuls[0]],
            per_channel=False,
            quant_last_matmul=False,
            op_types_to_quantize=["MatMul", "Gather"],
            extra_options={"WeightSymmetric": False, "ActivationSymmetric": True},
            execution_provider="CPUExecutionProvider",
        )
        quantization.quantize(self.model, "quant.onnx", cfg)
        q_model = onnx.load("quant.onnx")
        self.assertEqual(_count_op_num(q_model, "QLinearMatMul"), 1)

        # check nodes_to_exclude work
        cfg = config.StaticQuantConfig(
            calibration_data_reader=self.data_reader,
            weight_type=quantization.QuantType.QUInt8,
            nodes_to_exclude=[quantizable_matmuls[0]],
            per_channel=False,
            quant_last_matmul=False,
            extra_options={"WeightSymmetric": False, "ActivationSymmetric": True},
            execution_provider="CPUExecutionProvider",
        )
        quantization.quantize(self.model, "quant.onnx", cfg)
        q_model = onnx.load("quant.onnx")
        self.assertEqual(_count_op_num(q_model, "QLinearMatMul"), qmatmul_num_disable_last - 1)


    def test_dynamic_quant(self):
        cfg = config.DynamicQuantConfig(
            weight_type=quantization.QuantType.QInt8,
            per_channel=True,
            quant_last_matmul=False,
            extra_options={"WeightSymmetric": True, "ActivationSymmetric": False},
            execution_provider="CPUExecutionProvider",
        )
        quantization.quantize(self.model, "quant.onnx", cfg)

        cfg = config.DynamicQuantConfig(
            weight_type=quantization.QuantType.QUInt8,
            per_channel=False,
            quant_last_matmul=False,
            extra_options={"WeightSymmetric": False, "ActivationSymmetric": True},
            execution_provider="CPUExecutionProvider",
        )
        quantization.quantize(self.model, "quant.onnx", cfg, ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)



if __name__ == "__main__":
    unittest.main()
