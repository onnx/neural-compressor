import copy
import os
import shutil
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
import torch
import transformers
from optimum.exporters.onnx import main_export

from onnx_neural_compressor import data_reader, logger, onnx_model
from onnx_neural_compressor.quantization import algorithm_entry as algos
from onnx_neural_compressor.quantization import config, matmul_4bits_quantizer


def find_onnx_file(folder_path):
    # return first .onnx file path in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None


class DummyNLPDataloader(data_reader.CalibrationDataReader):

    def __init__(self, model_name, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        model = onnx.load(model_path, load_external_data=False)
        config = transformers.AutoConfig.from_pretrained(model_name)
        inputs_names = [input.name for input in model.graph.input]

        self.encoded_list = []
        encoded_input = dict(self.tokenizer(self.sequence_a, self.sequence_b, return_tensors="pt"))
        input_shape = encoded_input["input_ids"].shape
        encoded_input["position_ids"] = (
            torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
        )

        num_attention_heads = config.num_key_value_heads
        embed_size_per_head = config.hidden_size // config.num_attention_heads
        shape = (1, num_attention_heads, 0, embed_size_per_head)
        key_or_value = np.zeros(shape, dtype=np.float32)
        for input_name in inputs_names:
            if input_name not in encoded_input:
                encoded_input[input_name] = key_or_value

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


class TestLayerWiseQuant(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # onnx model exported with transformers>=4.38.0 is different with low version
        # which will cause layer-wise quant ut to fail
        # limit transformers to 4.37.2
        # TODO: remove transformers version limitation
        llama_id = "yujiepan/llama-2-tiny-3layers-random"
        main_export(llama_id, output="llama-2-tiny-3layers-random", task="text-generation-with-past")
        model_path = find_onnx_file("llama-2-tiny-3layers-random")
        self.llama = model_path

        model = onnx.load(model_path)
        model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        infer_shape_model_path = "llama-2-tiny-3layers-random/model-infer-shape.onnx"
        onnx.save(model, infer_shape_model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "llama-2-tiny-3layers-random/optimized_model.onnx"
        ort.InferenceSession(infer_shape_model_path, sess_options, providers=["CPUExecutionProvider"])

        self.llama_optimized = "llama-2-tiny-3layers-random/optimized_model.onnx"
        self.calibration_data_reader = DummyNLPDataloader(llama_id, self.llama_optimized)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("llama-2-tiny-3layers-random", ignore_errors=True)

    def setUp(self):
        # print the test name
        logger.info(f"Running ONNXRT TestLayerWiseQuant test: {self.id()}")

    def _check_model_is_quantized(self, model):
        node_optypes = [node.op_type for node in model.graph.node]
        return "MatMulNBits" in node_optypes or "MatMulFpQ4" in node_optypes

    def _get_quantized_matmul_weight(self, model, matmul_name):
        weight_init_name = None
        for node in model.graph.node:
            if node.name == matmul_name:
                weight_init_name = node.input[1]
        if weight_init_name is None:
            return None

        weight_init = None
        for init in model.graph.initializer:
            if init.name == weight_init_name:
                weight_init = onnx.numpy_helper.to_array(init)
        return weight_init

    def _apply_quantize(self, model, quant_config, quant_func, data_reader=None):
        fp32_model = copy.deepcopy(model)
        if data_reader is None:
            qmodel = quant_func(fp32_model, quant_config)
        else:
            qmodel = quant_func(fp32_model, quant_config, data_reader)
        self.assertIsNotNone(qmodel)
        return qmodel

    def test_rtn_layer_wise(self):
        # optimized model
        rtn_config = config.RTNConfig(layer_wise_quant=True)
        qmodel_lwq = self._apply_quantize(self.llama_optimized, rtn_config, algos.rtn_quantize_entry)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        rtn_config = config.RTNConfig(layer_wise_quant=False)
        qmodel = self._apply_quantize(self.llama_optimized, rtn_config, algos.rtn_quantize_entry)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())

        # original model
        rtn_config = config.RTNConfig(layer_wise_quant=True)
        qmodel_lwq = self._apply_quantize(self.llama, rtn_config, algos.rtn_quantize_entry)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        rtn_config = config.RTNConfig(layer_wise_quant=False)
        qmodel = self._apply_quantize(self.llama, rtn_config, algos.rtn_quantize_entry)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())

    def test_rtn_layer_wise_with_ort_like_api(self):
        # get qmodel without layer_wise_quant
        algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig(layer_wise_quant=False)
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.llama_optimized),
            algo_config=algo_config,
            optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )
        quant.process()
        qmodel = quant.model
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        # get qmodel with layer_wise_quant
        algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig(layer_wise_quant=True)
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.llama_optimized),
            algo_config=algo_config,
            optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )
        quant.process()
        qmodel_lwq = quant.model
        self.assertIsNotNone(qmodel_lwq)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        # compare qmodel
        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())

    def test_gptq_layer_wise(self):
        # optimized model
        self.calibration_data_reader.rewind()
        gptq_config = config.GPTQConfig(layer_wise_quant=True)
        qmodel_lwq = self._apply_quantize(
            self.llama_optimized, gptq_config, algos.gptq_quantize_entry, self.calibration_data_reader
        )
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        self.calibration_data_reader.rewind()
        gptq_config = config.GPTQConfig(layer_wise_quant=False)
        qmodel = self._apply_quantize(
            self.llama_optimized, gptq_config, algos.gptq_quantize_entry, self.calibration_data_reader
        )
        self.assertTrue(self._check_model_is_quantized(qmodel))

        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())

        # original model
        self.calibration_data_reader.rewind()
        gptq_config = config.GPTQConfig(layer_wise_quant=True)
        qmodel_lwq = self._apply_quantize(
            self.llama, gptq_config, algos.gptq_quantize_entry, self.calibration_data_reader
        )
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        self.calibration_data_reader.rewind()
        gptq_config = config.GPTQConfig(layer_wise_quant=False)
        qmodel = self._apply_quantize(self.llama, gptq_config, algos.gptq_quantize_entry, self.calibration_data_reader)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())

    def test_gptq_layer_wise_with_ort_like_api(self):
        # get qmodel without layer_wise_quant
        algo_config = matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig(
            layer_wise_quant=False, calibration_data_reader=self.calibration_data_reader
        )
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.llama_optimized),
            algo_config=algo_config,
            optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )
        quant.process()
        qmodel = quant.model
        self.assertIsNotNone(qmodel)
        self.assertTrue(self._check_model_is_quantized(qmodel))

        # get qmodel with layer_wise_quant
        algo_config = matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig(
            layer_wise_quant=True, calibration_data_reader=self.calibration_data_reader
        )
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            copy.deepcopy(self.llama_optimized),
            algo_config=algo_config,
            optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        )
        quant.process()
        qmodel_lwq = quant.model
        self.assertIsNotNone(qmodel_lwq)
        self.assertTrue(self._check_model_is_quantized(qmodel_lwq))

        # compare qmodel
        lwq_quantized_weight = self._get_quantized_matmul_weight(qmodel_lwq, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(lwq_quantized_weight)
        quantized_weight = self._get_quantized_matmul_weight(qmodel, "/lm_head/MatMul_Q4")
        self.assertIsNotNone(quantized_weight)
        self.assertTrue((lwq_quantized_weight == quantized_weight).all())


if __name__ == "__main__":
    unittest.main()
