"""Tests for algorithm utility components."""

import os
import unittest

import numpy as np
import onnx

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

    def test_make_woq_dq_node(self):
        node = onnx.helper.make_node("MatMul", ["input", "weight"], "output", name="Matmul")
        with self.assertRaises(ValueError):
            quant_utils.make_weight_only_dequant_node(
                node=node,
                weight_shape=(32, 32),
                block_size=16,
                num_bits=32,
                dtype="int",
                q_weight=np.random.randint(0, 10, size=(2, 32), dtype=np.uint8),
                scale=np.random.random((2, 32)),
                zero_point=np.zeros((2, 32)),
            )

    def test_split_shared_bias(self):
        input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 15, 15])
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 5, 11, 11])
        bias_initializer = onnx.numpy_helper.from_array(np.random.random(5).astype(np.float32), name="bias")
        conv1_weight_initializer = onnx.numpy_helper.from_array(
            np.random.randint(-1, 2, [5, 3, 3, 3]).astype(np.float32), name="conv1_weight"
        )
        conv1_node = onnx.helper.make_node("Conv", ["add_out", "conv1_weight", "bias"], ["conv1_output"], name="conv1")
        conv2_weight_initializer = onnx.numpy_helper.from_array(
            np.random.randint(-1, 2, [5, 5, 3, 3]).astype(np.float32), name="conv2_weight"
        )
        conv2_node = onnx.helper.make_node("Conv", ["add_out", "conv2_weight", "bias"], ["conv2_output"], name="conv2")
        initializers = [conv1_weight_initializer, conv2_weight_initializer, bias_initializer]
        graph = onnx.helper.make_graph([conv1_node, conv2_node], "test", [input], [output], initializer=initializers)
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        update_model = quant_utils.split_shared_bias(onnx_model.ONNXModel(model))
        split = any(["_nc_split_" in i.name for i in update_model.initializer()])
        self.assertTrue(split)

    def test_get_qmin_qmax_for_qType(self):
        with self.assertRaises(ValueError):
            quant_utils.get_qmin_qmax_for_qType(onnx.TensorProto.INT64)

        qmin, qmax = quant_utils.get_qmin_qmax_for_qType(onnx.TensorProto.INT8, reduce_range=True)
        self.assertEqual(qmin, -64)
        self.assertEqual(qmax, 64)


if __name__ == "__main__":
    unittest.main()
