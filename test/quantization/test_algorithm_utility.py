"""Tests for algorithm utility components."""

import os
import shutil
import unittest

import numpy as np
import onnx
import onnxruntime
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
import optimum.exporters.onnx

from onnx_neural_compressor import onnx_model
from onnx_neural_compressor.algorithms import utility as quant_utils


def find_onnx_file(folder_path):
    # return first .onnx file path in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None


class TestUtilityFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        llama_id = "yujiepan/llama-2-tiny-3layers-random"
        optimum.exporters.onnx.main_export(llama_id, output="llama-2-tiny-3layers-random", task="text-generation")
        model_path = find_onnx_file("llama-2-tiny-3layers-random")
        self.llama = model_path

        model = onnx.load(model_path)
        model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model, auto_merge=True)
        infer_shape_model_path = "llama-2-tiny-3layers-random/model-infer-shape.onnx"
        onnx.save(model, infer_shape_model_path)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "llama-2-tiny-3layers-random/optimized_model.onnx"
        onnxruntime.InferenceSession(infer_shape_model_path, sess_options)

        self.llama_optimized = "llama-2-tiny-3layers-random/optimized_model.onnx"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("llama-2-tiny-3layers-random", ignore_errors=True)

    def test_is_B_transposed(self):
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["y"],
            alpha=0.25,
            beta=0.35,
            transA=1,
            transB=1,
        )
        self.assertTrue(quant_utils.is_B_transposed(node))

        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["y"],
            alpha=0.25,
            beta=0.35,
        )
        self.assertFalse(quant_utils.is_B_transposed(node))

    def test_quantize_data(self):
        # sym int8
        data = [1, 2, 3, 4, 5]
        quantize_range = 127
        qType = onnx.onnx_pb.TensorProto.INT8
        scheme = "sym"
        rmin, rmax, zero_point, scale, quantized_data = quant_utils.quantize_data(data, quantize_range, qType, scheme)
        self.assertEqual(quantized_data.dtype, np.int8)

        scale, zero_point = quant_utils._calculate_scale_zp(np.array([0]), np.array([5]), quantize_range, qType, scheme)
        self.assertEqual(zero_point.dtype, np.int8)

        scale, zero_point = quant_utils._calculate_scale_zp(
            np.array([0]), np.array([127]), quantize_range, qType, scheme
        )
        self.assertEqual(zero_point.dtype, np.int8)

        # asym uint8
        data = [-1, 0, 1, 2, 3]
        quantize_range = 255
        qType = onnx.onnx_pb.TensorProto.UINT8
        scheme = "asym"
        rmin, rmax, zero_point, scale, quantized_data = quant_utils.quantize_data(data, quantize_range, qType, scheme)
        self.assertEqual(quantized_data.dtype, np.uint8)

        scale, zero_point = quant_utils._calculate_scale_zp(np.array([0]), np.array([5]), quantize_range, qType, scheme)
        self.assertEqual(zero_point.dtype, np.uint8)

        scale, zero_point = quant_utils._calculate_scale_zp(
            np.array([0]), np.array([255]), quantize_range, qType, scheme
        )
        self.assertEqual(zero_point.dtype, np.uint8)

        # unexpected combination
        with self.assertRaises(ValueError) as cm:
            rmin, rmax, zero_point, scale, quantized_data = quant_utils.quantize_data(
                data, quantize_range, qType=onnx.onnx_pb.TensorProto.UINT8, scheme="sym"
            )
        self.assertTrue("Unexpected combination of data type" in str(cm.exception))

    def test_get_qrange_for_qType(self):
        qrange = quant_utils.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.UINT8)
        self.assertEqual(qrange, 255)
        qrange = quant_utils.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.UINT8, reduce_range=True)
        self.assertEqual(qrange, 127)
        qrange = quant_utils.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.INT8)
        self.assertEqual(qrange, 254)
        qrange = quant_utils.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.INT8, reduce_range=True)
        self.assertEqual(qrange, 128)

        # unexpected quantization data type
        with self.assertRaises(ValueError) as cm:
            quant_utils.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.FLOAT)
        self.assertEqual(str(cm.exception), "unsupported quantization data type")

    def test_check_model_with_infer_shapes(self):
        self.assertFalse(quant_utils.check_model_with_infer_shapes(self.llama))
        self.assertTrue(quant_utils.check_model_with_infer_shapes(self.llama_optimized))
        self.assertTrue(
            quant_utils.check_model_with_infer_shapes(
                onnx_model.ONNXModel(onnx.load(self.llama_optimized, load_external_data=False))
            )
        )
