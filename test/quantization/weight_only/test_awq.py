import copy
import itertools
import os
import shutil
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from optimum.exporters.onnx import main_export
from packaging import version

from onnx_neural_compressor import data_reader, logger
from onnx_neural_compressor.quantization import QuantFormat
from onnx_neural_compressor.quantization import algorithm_entry as algos
from onnx_neural_compressor.quantization import config, matmul_4bits_quantizer, matmul_nbits_quantizer


def find_onnx_file(folder_path):
    # return first .onnx file path in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None


class DummyNLPDataloader(data_reader.CalibrationDataReader):

    def __init__(self, model_name):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"

        self.encoded_list = []
        encoded_input = dict(self.tokenizer(self.sequence_a, self.sequence_b, return_tensors="pt"))
        input_shape = encoded_input["input_ids"].shape
        encoded_input["position_ids"] = (
            torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        )

        # convert torch tensor to numpy
        for input_name, input_value in encoded_input.items():
            if isinstance(input_value, torch.Tensor):
                encoded_input[input_name] = input_value.numpy()

        self.encoded_list.append(encoded_input)
        self.iter_next = iter(self.encoded_list)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.encoded_list)


class MatMulDataloader(data_reader.CalibrationDataReader):

    def __init__(self):
        self.encoded_list = [{"A": torch.randn((1, 11008)).numpy()}]
        self.iter_next = iter(self.encoded_list)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.encoded_list)


def build_matmul_model():
    # MatMul - Add - Add
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 11008])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1, 1024])
    D = onnx.helper.make_tensor_value_info("D", onnx.TensorProto.FLOAT, [1, 1024])

    B_init = onnx.helper.make_tensor("B", onnx.TensorProto.FLOAT, [11008, 1024], np.random.random((11008, 1024)))
    E_init = onnx.helper.make_tensor("E", onnx.TensorProto.FLOAT, [1, 1024], np.random.random((1, 1024)))

    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    add = onnx.helper.make_node("Add", ["C", "E"], ["D"], name="add")

    graph = onnx.helper.make_graph([matmul_node, add], "test_graph_1", [A], [D], [B_init, E_init])
    model = onnx.helper.make_model(graph)
    model = onnx.helper.make_model(graph, **{"opset_imports": [onnx.helper.make_opsetid("", 13)]})
    return model


class TestAWQQuant(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="gptj",
        )
        self.gptj = find_onnx_file("./gptj")
        self.calibration_data_reader = DummyNLPDataloader("hf-internal-testing/tiny-random-gptj")

        self.matmul_model = build_matmul_model()
        self.matmul_data_reader = MatMulDataloader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("gptj", ignore_errors=True)

    def setUp(self):
        # print the test name
        logger.info(f"Running ONNXRT TestAWQQuant test: {self.id()}")

    def _count_woq_matmul(self, q_model, bits=4, group_size=32):
        op_names = [
            i.name
            for i in q_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(bits, group_size))
        ]
        return len(op_names)

    def _check_model_is_quantized(self, model):
        node_optypes = [node.op_type for node in model.graph.node]
        return "MatMulNBits" in node_optypes or "MatMulFpQ4" in node_optypes

    def _check_node_is_quantized(self, model, node_name):
        for node in model.graph.node:
            if (node.name == node_name or node.name == node_name + "_Q4") and node.op_type in [
                "MatMulNBits",
                "MatMulFpQ4",
            ]:
                return True
        return False

    def _apply_awq(self, quant_config):
        logger.info(f"Test AWQ with config {quant_config}")

        fp32_model = copy.deepcopy(self.gptj)
        qmodel = algos.awq_quantize_entry(
            fp32_model, quant_config, calibration_data_reader=self.calibration_data_reader
        )
        self.assertIsNotNone(qmodel)
        return qmodel


class TestAWQQuantWithInternalAPI(TestAWQQuant):

    def test_awq_params_combination(self):

        # some tests were skipped to accelerate the CI
        # TODO: check params combination.
        # TODO: Add number check for group_size.
        awq_options = {
            "weight_dtype": ["int"],
            "weight_bits": [4, 3, 8],
            "weight_group_size": [32],
            "weight_sym": [True, False],
            "act_dtype": ["fp32"],
            "accuracy_level": [0],
            "quant_format": [0, 1],
            "enable_auto_scale": [True, False],
            "enable_mse_search": [True, False],
        }

        keys = config.AWQConfig.params_list + config.AWQConfig.model_params_list
        for value in itertools.product(*awq_options.values()):
            d = dict(zip(keys, value))
            print(d)
            quant_config = config.AWQConfig(**d)
            qmodel = self._apply_awq(quant_config)
            self.assertEqual(self._count_woq_matmul(qmodel, bits=value[1], group_size=value[2]), 30)

    def test_awq_config(self):

        awq_config1 = config.AWQConfig(weight_bits=4)
        quant_config_dict = {
            "awq": {"weight_bits": 4},
        }
        awq_config2 = config.AWQConfig.from_dict(quant_config_dict["awq"])
        self.assertEqual(awq_config1.to_dict(), awq_config2.to_dict())

        tuning_config = config.AWQConfig.get_config_set_for_tuning()
        self.assertTrue(isinstance(tuning_config, config.AWQConfig))

    def test_quantize_awq_from_dict_default(self):

        qmodel = self._apply_awq(quant_config=config.get_default_awq_config())
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

    def test_quantize_awq_from_class_beginner(self):

        quant_config = config.AWQConfig(weight_bits=4, weight_group_size=32)
        qmodel = self._apply_awq(quant_config)
        self.assertIsNotNone(qmodel)

    def test_quantize_awq_fallback(self):

        quant_config = config.AWQConfig(
            weight_dtype="int",
            weight_sym=False,
            weight_group_size=32,
            nodes_to_exclude=["/h.4/mlp/fc_out/MatMul"],
        )
        qmodel = self._apply_awq(quant_config)
        self.assertIsNotNone(qmodel)
        self.assertEqual(self._count_woq_matmul(qmodel), 29)
        self.assertFalse(self._check_node_is_quantized(qmodel, "/h.4/mlp/fc_out/MatMul"))

        # test quant_last_matmul
        quant_config = config.AWQConfig(
            weight_dtype="int",
            weight_sym=False,
            weight_group_size=32,
            quant_last_matmul=False,
        )
        qmodel = self._apply_awq(quant_config)
        self.assertIsNotNone(qmodel)
        self.assertEqual(self._count_woq_matmul(qmodel), 29)
        self.assertFalse(self._check_node_is_quantized(qmodel, "/h.4/mlp/fc_out/MatMul"))

    @unittest.skipIf(
        version.Version(ort.__version__) < version.Version("1.19.0"),
        "Please use onnxruntime >= 1.19.0 for QDQ format test",
    )
    def test_awq_with_QDQ_format(self):

        quant_config = config.AWQConfig(
            weight_dtype="int",
            weight_sym=False,
            weight_group_size=32,
            weight_bits=4,
            quant_format=QuantFormat.QDQ,
        )

        op21_model = copy.deepcopy(self.matmul_model)
        op21_model.opset_import[0].version = 21
        qmodel = algos.awq_quantize_entry(op21_model, quant_config, calibration_data_reader=self.matmul_data_reader)

        self.assertIsNotNone(qmodel)
        self.assertTrue("MatMul" in [i.op_type for i in qmodel.graph.node])
        self.assertTrue("DequantizeLinear" in [i.op_type for i in qmodel.graph.node])


class TestAWQQuantWithORTLikeAPI(TestAWQQuant):

    def test_awq_config_4bits(self):
        algo_config = matmul_4bits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.calibration_data_reader
        )

        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.gptj),
            block_size=32,
            is_symmetric=False,
            algo_config=algo_config,
        )
        quant.process()
        self.assertIsNotNone(quant.model)
        self.assertTrue(self._check_model_is_quantized(quant.model))

    def test_awq_config_4bits_with_accuracy_level(self):
        algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig()

        for accuracy_level in [0, 1, 2, 3, 4]:
            quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
                copy.deepcopy(self.gptj),
                block_size=32,
                is_symmetric=False,
                algo_config=algo_config,
                accuracy_level=accuracy_level,
            )
            quant.process()

            for node in quant.model.graph.node:
                if node.op_type == "MatMulNBits":
                    for attr in node.attribute:
                        if attr.name == "accuracy_level":
                            self.assertEqual(attr.i, accuracy_level)
                            break
            self.assertIsNotNone(quant.model)
            self.assertTrue(self._check_model_is_quantized(quant.model))

    def test_awq_config_4bits_with_exclude_node(self):

        algo_config = matmul_4bits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.calibration_data_reader
        )

        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.gptj),
            block_size=32,
            is_symmetric=False,
            algo_config=algo_config,
            nodes_to_exclude=["/h.4/mlp/fc_out/MatMul"],
        )
        quant.process()
        self.assertIsNotNone(quant.model)
        self.assertTrue(self._check_model_is_quantized(quant.model))
        self.assertFalse(self._check_node_is_quantized(quant.model, "/h.4/mlp/fc_out/MatMul"))

    def test_awq_config_nbits(self):

        algo_config = matmul_nbits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.calibration_data_reader
        )

        for n_bits in [3, 4, 8]:
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                copy.deepcopy(self.gptj),
                n_bits=n_bits,
                block_size=32,
                is_symmetric=False,
                algo_config=algo_config,
            )
            quant.process()
            self.assertIsNotNone(quant.model)
            self.assertEqual(self._count_woq_matmul(quant.model, bits=n_bits, group_size=32), 30)

    def test_awq_config_nbits_with_exclude_node(self):

        algo_config = matmul_nbits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.calibration_data_reader
        )

        for n_bits in [3, 4, 8]:
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                copy.deepcopy(self.gptj),
                n_bits=n_bits,
                block_size=32,
                is_symmetric=False,
                algo_config=algo_config,
                nodes_to_exclude=["/h.4/mlp/fc_out/MatMul"],
            )
            quant.process()
            self.assertIsNotNone(quant.model)
            self.assertEqual(self._count_woq_matmul(quant.model, bits=n_bits, group_size=32), 29)

    def test_awq_with_specified_matmul(self):

        algo_config = matmul_nbits_quantizer.AWQWeightOnlyQuantConfig(calibration_data_reader=self.matmul_data_reader)

        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            copy.deepcopy(self.matmul_model),
            n_bits=4,
            block_size=32,
            is_symmetric=False,
            algo_config=algo_config,
            optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )
        quant.process()
        self.assertIsNotNone(quant.model)
        self.assertEqual(self._count_woq_matmul(quant.model, bits=4, group_size=32), 1)

    @unittest.skipIf(
        version.Version(ort.__version__) < version.Version("1.19.0"),
        "Please use onnxruntime >= 1.19.0 for QDQ format test",
    )
    def test_awq_with_QDQ_format(self):

        algo_config = matmul_nbits_quantizer.AWQWeightOnlyQuantConfig(
            calibration_data_reader=self.matmul_data_reader, quant_format=QuantFormat.QDQ
        )

        op21_model = copy.deepcopy(self.matmul_model)
        op21_model.opset_import[0].version = 21

        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            op21_model,
            n_bits=4,
            block_size=32,
            is_symmetric=False,
            algo_config=algo_config,
            optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )
        quant.process()
        self.assertIsNotNone(quant.model)
        self.assertTrue("MatMul" in [i.op_type for i in quant.model.graph.node])
        self.assertTrue("DequantizeLinear" in [i.op_type for i in quant.model.graph.node])


if __name__ == "__main__":
    unittest.main()
