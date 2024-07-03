"""Tests for algorithm utility components."""

import os
import unittest

import numpy as np
import onnx

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
