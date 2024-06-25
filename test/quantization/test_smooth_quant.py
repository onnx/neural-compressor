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

import glob
import os
import shutil
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from optimum.exporters.onnx import main_export

from onnx_neural_compressor import data_reader
from onnx_neural_compressor.quantization import config, quantize, QuantType
from onnx_neural_compressor.quantization import algorithm_entry as algos


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


class TestONNXRT3xSmoothQuant(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="gptj",
        )
        self.gptj = glob.glob(os.path.join("./gptj", "*.onnx"))[0]
        self.data_reader = DataReader(self.gptj)
        self.quant_gptj = os.path.join("./gptj", "quant_model.onnx")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./gptj", ignore_errors=True)
        os.remove("Optimized_model.onnx")

    def test_sq_from_class_beginner(self):
        self.data_reader.rewind()
        sq_config = config.get_default_sq_config()
        model = algos.smooth_quant_entry(self.gptj, sq_config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 30)

    def test_sq_auto_tune_from_class_beginner(self):
        self.data_reader.rewind()
        sq_config = config.SmoothQuantConfig(alpha="auto", scales_per_op=False)
        model = algos.smooth_quant_entry(self.gptj, sq_config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 15)

    def test_sq_ort_param_class_beginner(self):
        self.data_reader.rewind()
        sq_config = config.SmoothQuantConfig(weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8)
        model = algos.smooth_quant_entry(self.gptj, sq_config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertTrue(2 in [i.data_type for i in model.graph.initializer])
        self.assertTrue(3 not in [i.data_type for i in model.graph.initializer])
        self.assertEqual(num_muls, 30)

    def test_sq_with_ort_like_api(self):
        self.data_reader.rewind()
        sq_config = config.StaticQuantConfig(
            self.data_reader,
            weight_type=QuantType.QUInt8,
            activation_type=QuantType.QUInt8,
            extra_options={"SmoothQuant": True, "SmoothQuantAlpha": 0.7, "SmoothQuantCalibIter": 1},
        )
        quantize(self.gptj, self.quant_gptj, sq_config)
        model = onnx.load(self.quant_gptj)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertTrue(2 in [i.data_type for i in model.graph.initializer])
        self.assertTrue(3 not in [i.data_type for i in model.graph.initializer])
        self.assertEqual(num_muls, 30)

    def test_smooth_quant_args(self):
        self.data_reader.rewind()
        sq_config = config.SmoothQuantConfig(
            weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8, alpha="auto"
        )
        model = algos.smooth_quant_entry(self.gptj, sq_config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 30)

        self.data_reader.rewind()
        sq_config = config.SmoothQuantConfig(
            weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8, scales_per_op=False
        )
        model = algos.smooth_quant_entry(self.gptj, sq_config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 15)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "Optimized_model.onnx"
        sess = ort.InferenceSession(self.gptj, sess_options, providers=["CPUExecutionProvider"])
        self.data_reader.rewind()
        sq_config = config.SmoothQuantConfig(
            weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8, folding=True, scales_per_op=False
        )
        model = algos.smooth_quant_entry("Optimized_model.onnx", sq_config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 10)

        self.data_reader.rewind()
        sq_config = config.SmoothQuantConfig(
            weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8, folding=False, scales_per_op=False
        )
        model = algos.smooth_quant_entry("Optimized_model.onnx", sq_config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 15)


if __name__ == "__main__":
    unittest.main()
