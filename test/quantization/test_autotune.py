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
from onnx_neural_compressor.quantization import config, tuning

from typing import Callable, Dict, List, Optional, Union  # isort: skip


def fake_eval(model, eval_result_lst):
    acc = eval_result_lst.pop(0)
    return acc


def _create_evaluator_for_eval_fns(eval_fns: Optional[Union[Callable, Dict, List[Dict]]] = None) -> tuning.Evaluator:
    evaluator = tuning.Evaluator()
    evaluator.set_eval_fn_registry(eval_fns)
    return evaluator


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


class TestONNXRTAutoTune(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="gptj",
        )
        self.gptj = glob.glob(os.path.join("./gptj", "*.onnx"))[0]
        self.data_reader = DataReader(self.gptj)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./gptj", ignore_errors=True)

    @mock.patch("onnx_neural_compressor.logger.warning")
    def test_auto_tune_warning(self, mock_warning):
        acc_data = iter([1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        def eval_acc_fn(model) -> float:
            session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            return next(acc_data)

        custom_tune_config = tuning.TuningConfig(
            config_set=[config.SmoothQuantConfig(alpha=0.5), config.SmoothQuantConfig(alpha=0.6)]
        )
        with self.assertRaises(SystemExit):
            best_model = tuning.autotune(
                model_input=self.gptj,
                tune_config=custom_tune_config,
                eval_fn=eval_acc_fn,
                calibration_data_reader=self.data_reader,
            )
        call_args_list = mock_warning.call_args_list
        # There may be multiple calls to warning, so we need to check all of them
        self.assertIn(
            "Please refine your eval_fn to accept model path (str) as input.", [info[0][0] for info in call_args_list]
        )

    def test_sq_auto_tune(self):
        acc_data = iter([1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        perf_data = iter([1.0, 0.9, 0.99])

        def eval_perf_fn(model) -> float:
            return next(perf_data)

        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]

        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = tuning.TuningConfig(
            config_set=[config.SmoothQuantConfig(alpha=0.5), config.SmoothQuantConfig(alpha=0.6)]
        )
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_acc_fn,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNotNone(best_model)

        custom_tune_config = tuning.TuningConfig(config_set=[config.SmoothQuantConfig(alpha=[0.5, 0.6])])
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calibration_data_reader=self.data_reader,
        )
        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)

    def test_rtn_auto_tune(self):
        eval_acc_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.9])
        with self.assertRaises(SystemExit):
            custom_tune_config = tuning.TuningConfig(
                config_set=[config.RTNConfig(weight_group_size=32), config.RTNConfig(weight_group_size=64)]
            )
            best_model = tuning.autotune(
                model_input=self.gptj,
                tune_config=custom_tune_config,
                eval_fn=eval_acc_fn,
                calibration_data_reader=self.data_reader,
            )

        eval_perf_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.99, 0.99])
        eval_acc_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.99])
        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]
        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = tuning.TuningConfig(config_set=[config.RTNConfig(weight_group_size=[32, 64])])
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calibration_data_reader=self.data_reader,
        )

        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)

        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(4, 64))
        ]
        self.assertTrue(len(op_names) > 0)

    def test_awq_auto_tune(self):
        eval_acc_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.9])
        with self.assertRaises(SystemExit):
            custom_tune_config = tuning.TuningConfig(
                config_set=[config.AWQConfig(weight_group_size=32), config.AWQConfig(weight_group_size=64)]
            )
            best_model = tuning.autotune(
                model_input=self.gptj,
                tune_config=custom_tune_config,
                eval_fn=eval_acc_fn,
                calibration_data_reader=self.data_reader,
            )

        eval_perf_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.99, 0.99])
        eval_acc_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.99, 0.99])
        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]
        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = tuning.TuningConfig(config_set=[config.AWQConfig(weight_group_size=[32, 64])])
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calibration_data_reader=self.data_reader,
        )
        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)
        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(4, 32))
        ]
        self.assertTrue(len(op_names) > 0)

    def test_gptq_auto_tune(self):
        eval_acc_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.9])
        with self.assertRaises(SystemExit):
            custom_tune_config = tuning.TuningConfig(
                config_set=[config.GPTQConfig(weight_group_size=32), config.GPTQConfig(weight_group_size=64)]
            )
            best_model = tuning.autotune(
                model_input=self.gptj,
                tune_config=custom_tune_config,
                eval_fn=eval_acc_fn,
                calibration_data_reader=self.data_reader,
            )

        eval_perf_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.99, 0.99])
        eval_acc_fn = functools.partial(fake_eval, eval_result_lst=[1.0, 0.99, 0.99])
        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]
        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = tuning.TuningConfig(config_set=[config.GPTQConfig(weight_group_size=[32, 64])])
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calibration_data_reader=self.data_reader,
        )
        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)
        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(4, 32))
        ]
        self.assertTrue(len(op_names) > 0)

    def test_woq_auto_tune(self):
        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        custom_tune_config = tuning.TuningConfig(
            config_set=[config.RTNConfig(weight_bits=4), config.AWQConfig(weight_bits=8)]
        )
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=partial_fake_eval,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNotNone(best_model)
        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(8, 32))
        ]
        self.assertTrue(len(op_names) > 0)
        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.81, 1.0, 0.99, 0.99])

        custom_tune_config = tuning.TuningConfig(config_set=config.get_woq_tuning_config())
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=partial_fake_eval,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNotNone(best_model)
        self.assertEqual(
            len(op_names),
            len(
                [
                    i.name
                    for i in best_model.graph.node
                    if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(4, 32))
                ]
            )
            + 1,
        )

        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.81, 0.81, 1.0, 0.99])

        custom_tune_config = tuning.TuningConfig(config_set=config.get_woq_tuning_config())
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=partial_fake_eval,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNotNone(best_model)
        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(4, 128))
        ]
        self.assertTrue(len(op_names) > 0)

    def test_dynamic_auto_tune(self):
        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.99, 0.81, 1.0, 0.99])

        custom_tune_config = tuning.TuningConfig(config_set=config.DynamicQuantConfig.get_config_set_for_tuning())
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=partial_fake_eval,
        )
        self.assertIsNotNone(best_model)

    def test_dynamic_custom_auto_tune(self):
        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.99])
        custom_tune_config = tuning.TuningConfig(
            config_set=config.DynamicQuantConfig(
                per_channel=[True, False],
                execution_provider="CPUExecutionProvider",
            )
        )
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=partial_fake_eval,
            calibration_data_reader=self.data_reader,
        )

        optypes = [i.op_type for i in best_model.graph.node]
        self.assertTrue("DynamicQuantizeLinear" in optypes)
        self.assertTrue("MatMulInteger" in optypes)
        ort.InferenceSession(best_model.SerializeToString(), providers=["CPUExecutionProvider"])
        self.assertIsNotNone(best_model)

        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.82, 0.81, 1.0, 0.99])
        for execution_provider in ["DmlExecutionProvider", "TensorrtExecutionProvider"]:
            with self.assertRaises(SystemExit):
                custom_tune_config = tuning.TuningConfig(
                    config_set=config.DynamicQuantConfig(
                        per_channel=[True, False],
                        execution_provider=execution_provider,
                    )
                )
                best_model = tuning.autotune(
                    model_input=self.gptj,
                    tune_config=custom_tune_config,
                    eval_fn=partial_fake_eval,
                    calibration_data_reader=self.data_reader,
                )

    def test_static_default_auto_tune(self):
        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.99])

        custom_tune_config = tuning.TuningConfig(
            config_set=config.StaticQuantConfig.get_config_set_for_tuning(
                execution_provider="TensorrtExecutionProvider",
                quant_format=quantization.QuantFormat.QDQ,
            )
        )
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=partial_fake_eval,
            calibration_data_reader=self.data_reader,
        )
        optypes = [i.op_type for i in best_model.graph.node]
        self.assertTrue("QLinearMatMul" not in optypes)
        self.assertTrue("QuantizeLinear" in optypes)
        self.assertTrue("MatMul" in optypes)
        ort.InferenceSession(best_model.SerializeToString(), providers=["TensorrtExecutionProvider"])
        self.assertIsNotNone(best_model)

    def test_static_custom_auto_tune(self):
        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.99])

        custom_tune_config = tuning.TuningConfig(
            config_set=config.StaticQuantConfig(
                per_channel=[True, False],
                execution_provider="CPUExecutionProvider",
                quant_format=quantization.QuantFormat.QOperator,
            )
        )
        best_model = tuning.autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=partial_fake_eval,
            calibration_data_reader=self.data_reader,
        )

        optypes = [i.op_type for i in best_model.graph.node]
        self.assertTrue("QLinearMatMul" in optypes)
        self.assertTrue("QuantizeLinear" in optypes)
        ort.InferenceSession(best_model.SerializeToString(), providers=["CPUExecutionProvider"])
        self.assertIsNotNone(best_model)

    @mock.patch("onnx_neural_compressor.logger.warning")
    def test_skip_verified_config_mapping(self, mock_warning):
        partial_fake_eval = functools.partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.99])

        with self.assertRaises(SystemExit):
            custom_tune_config = tuning.TuningConfig(
                config_set=config.StaticQuantConfig(
                    per_channel=[True, False],
                    execution_provider="DmlExecutionProvider",
                )
            )
            best_model = tuning.autotune(
                model_input=self.gptj,
                tune_config=custom_tune_config,
                eval_fn=partial_fake_eval,
                calibration_data_reader=self.data_reader,
            )
        call_args_list = mock_warning.call_args_list
        # There may be multiple calls to warning, so we need to check all of them
        self.assertIn("Skip the verified config mapping.", [info[0][0] for info in call_args_list])


if __name__ == "__main__":
    unittest.main()
