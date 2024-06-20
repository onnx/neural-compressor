import os
import shutil
import sys
import unittest

import numpy as np
import onnx

from onnx_neural_compressor import data_reader
from onnx_neural_compressor.algorithms.post_training_quant import calibrate, calibrator


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """Helper function to generate initializers for test inputs."""
    tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
    init = onnx.numpy_helper.from_array(tensor, input_name)
    return init


class DataReader(data_reader.CalibrationDataReader):

    def __init__(self):
        self.data_list = []
        self.data_list.append(
            {
                "input0": np.array([[[[0.45, 0.60, 0.75]], [[0.25, 0.50, 0.75]], [[0.90, 0.70, 0.50]]]]).astype(
                    np.float32
                )
            }
        )
        self.data_list.append(
            {
                "input0": np.array([[[[0.62, 0.94, 0.38]], [[0.70, 0.13, 0.07]], [[0.89, 0.75, 0.84]]]]).astype(
                    np.float32
                )
            }
        )
        self.data_list.append(
            {
                "input0": np.array([[[[0.64, 0.24, 0.97]], [[0.82, 0.58, 0.27]], [[0.019, 0.34, 0.02]]]]).astype(
                    np.float32
                )
            }
        )
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class DataReader2(data_reader.CalibrationDataReader):

    def __init__(self):
        self.data_list = []
        self.data_list.append({"A": np.random.random([1, 1, 5, 5]).astype(np.float32)})
        self.data_list.append({"A": np.random.random([1, 1, 5, 5]).astype(np.float32)})
        self.data_list.append({"A": np.random.random([1, 1, 5, 5]).astype(np.float32)})
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def create_cv_session():
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [1, 1, 3, 3])
    b_value = np.random.randn(1, 1, 3, 3).astype(np.float32)
    B_init = onnx.helper.make_tensor("B", onnx.TensorProto.FLOAT, [1, 1, 3, 3], b_value.reshape(9).tolist())
    D = onnx.helper.make_tensor_value_info("D", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
    conv_node = onnx.helper.make_node("Conv", ["A", "B"], ["C"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    relu_node = onnx.helper.make_node("Relu", ["C"], ["D"], name="relu")
    graph = onnx.helper.make_graph([conv_node, relu_node], "test_graph_1", [A], [D], [B_init])
    model = onnx.helper.make_model(graph, **{"opset_imports": [onnx.helper.make_opsetid("", 13)]})
    dataloader = DataReader2()
    return model, dataloader


class TestCalibrate(unittest.TestCase):
    work_space = "./onnxrt_calib_test"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.work_space)
        cls.cv_session = create_cv_session()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_space, ignore_errors=True)

    def test_dump_calibration(self):
        model, dataloader = self.cv_session
        augment = calibrate.ONNXRTAugment(model, dataloader, ["Conv", "Relu"], iterations=[0])
        calib_params = augment.dump_calibration({})
        self.assertTrue("A" in calib_params and "B" in calib_params and "D" in calib_params and "C" in calib_params)

    def test_augment_graph(self):
        """TEST_CONFIG_1."""

        #     Conv
        #      |
        #     Clip
        #      |
        #     MatMul

        A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [1, 1, 3, 3])
        E = onnx.helper.make_tensor_value_info("E", onnx.TensorProto.FLOAT, [1, 1, 5, 1])
        F = onnx.helper.make_tensor_value_info("F", onnx.TensorProto.FLOAT, [1, 1, 5, 1])
        conv_node = onnx.helper.make_node(
            "Conv", ["A", "B"], ["C"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        clip_node = onnx.helper.make_node("Clip", ["C"], ["D"], name="Clip")
        matmul_node = onnx.helper.make_node("MatMul", ["D", "E"], ["F"], name="MatMul")
        graph = onnx.helper.make_graph([conv_node, clip_node, matmul_node], "test_graph_1", [A, B, E], [F])
        model = onnx.helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = calibrate.ONNXRTAugment(model, data_reader, ["Conv", "MatMul"])
        augment.augment_graph()
        augmented_model = augment.augmented_model

        # Checking if output exists
        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["Conv", "Clip", "MatMul"]
        added_outputs = ["A", "B", "C", "D", "E", "F"]
        # Original 3 nodes (exclude graph input/output)
        self.assertEqual(len(augmented_model_node_names), 3)
        # Original 1 graph output + 5 intermediate outputs
        self.assertEqual(len(augmented_model_outputs), 6)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_1")
        """TEST_CONFIG_2."""

        #   Conv
        #    |
        #   Conv

        G = onnx.helper.make_tensor_value_info("G", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        H = onnx.helper.make_tensor_value_info("H", onnx.TensorProto.FLOAT, [1, 1, 3, 3])
        J = onnx.helper.make_tensor_value_info("J", onnx.TensorProto.FLOAT, [1, 1, 3, 3])
        K = onnx.helper.make_tensor_value_info("K", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node_1 = onnx.helper.make_node(
            "Conv", ["G", "H"], ["I"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        conv_node_2 = onnx.helper.make_node(
            "Conv", ["I", "J"], ["K"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        graph = onnx.helper.make_graph([conv_node_1, conv_node_2], "test_graph_2", [G, H, J], [K])
        model = onnx.helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = calibrate.ONNXRTAugment(
            model,
            data_reader,
            ["Conv", "MatMul"],
        )
        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["Conv", "Conv"]
        added_outputs = ["I", "J", "H", "G", "K"]
        # Original 2 nodes
        self.assertEqual(len(augmented_model_node_names), 2)
        # Original 1 graph output + 4 intermediate outputs
        self.assertEqual(len(augmented_model_outputs), 5)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_2")
        """TEST_CONFIG_3."""

        #   Relu
        #    |
        #   Conv  \
        #    |     |
        #   Clip   |
        #    |    /
        #   MatMul

        L = onnx.helper.make_tensor_value_info("L", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        N = onnx.helper.make_tensor_value_info("N", onnx.TensorProto.FLOAT, [1, 1, 3, 3])
        Q = onnx.helper.make_tensor_value_info("Q", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        relu_node = onnx.helper.make_node("Relu", ["L"], ["M"], name="Relu")
        conv_node = onnx.helper.make_node(
            "Conv", ["M", "N"], ["O"], name="Conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )
        clip_node = onnx.helper.make_node("Clip", ["O"], ["P"], name="Clip")
        matmul_node = onnx.helper.make_node("MatMul", ["P", "M"], ["Q"], name="MatMul")
        graph = onnx.helper.make_graph([relu_node, conv_node, clip_node, matmul_node], "test_graph_3", [L, N], [Q])
        model = onnx.helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = calibrate.ONNXRTAugment(model, data_reader, ["Conv", "MatMul"])
        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["Relu", "Conv", "Clip", "MatMul"]
        added_outputs = ["P", "M", "N", "O", "Q"]
        # Original 4 nodes
        self.assertEqual(len(augmented_model_node_names), 4)
        # Original 1 graph output + 4 intermediate outputs
        self.assertEqual(len(augmented_model_outputs), 5)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_3")
        """TEST_CONFIG_4."""

        #   Attention
        #    |
        #   MatMul

        Attention_weight = onnx.helper.make_tensor_value_info("Attention_weight", onnx.TensorProto.FLOAT, [13, 7])
        Attention_bias = onnx.helper.make_tensor_value_info("Attention_bias", onnx.TensorProto.FLOAT, [13, 7])
        Attention_mask = onnx.helper.make_tensor_value_info("Attention_mask", onnx.TensorProto.INT32, [13, 7])
        S = onnx.helper.make_tensor_value_info("S", onnx.TensorProto.FLOAT, [13, 7])
        T = onnx.helper.make_tensor_value_info("T", onnx.TensorProto.FLOAT, [13, 7])
        attention_node = onnx.helper.make_node(
            "Attention", ["Attention_weight", "Attention_bias", "Attention_mask"], ["R"], name="Attention"
        )
        matmul_node = onnx.helper.make_node("MatMul", ["R", "S"], ["T"], name="MatMul")
        graph = onnx.helper.make_graph(
            [attention_node, matmul_node], "test_graph_4", [Attention_weight, Attention_bias, Attention_mask, S], [T]
        )
        model = onnx.helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = calibrate.ONNXRTAugment(model, data_reader, ["Conv", "MatMul", "Attention"])
        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["Attention", "MatMul"]
        added_outputs = ["R", "Attention_mask", "S", "T", "Attention_bias", "Attention_weight"]
        # Original 2 nodes
        self.assertEqual(len(augmented_model_node_names), 2)
        # Original 1 graph output + 5 intermediate outputs
        self.assertEqual(len(augmented_model_outputs), 6)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_4")

        #    QAttention
        #        |
        #    QuantizeLinear

        Attention_input = onnx.helper.make_tensor_value_info("input_quantized", onnx.TensorProto.INT8, [7, 13])
        Attention_weight = onnx.helper.make_tensor_value_info("weight_quantized", onnx.TensorProto.INT8, [13, 7])
        weight_quantized = generate_input_initializer([13, 7], np.int8, "weight_quantized")
        Attention_bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [13, 7])
        bias = generate_input_initializer([13, 7], np.float32, "bias")
        Input_scale = onnx.helper.make_tensor_value_info("input_scale", onnx.TensorProto.FLOAT, [1])
        input_scale = generate_input_initializer([1], np.float32, "input_scale")
        Weight_scale = onnx.helper.make_tensor_value_info("weight_scale", onnx.TensorProto.FLOAT, [1])
        weight_scale = generate_input_initializer([1], np.float32, "weight_scale")
        Attention_mask = onnx.helper.make_tensor_value_info("mask", onnx.TensorProto.INT32, [13, 7])
        mask = generate_input_initializer([13, 7], np.int32, "mask")
        Input_zo = onnx.helper.make_tensor_value_info("input_zero_point", onnx.TensorProto.INT8, [1])
        input_zero_point = generate_input_initializer([1], np.int8, "input_zero_point")
        Weight_zo = onnx.helper.make_tensor_value_info("weight_zero_point", onnx.TensorProto.INT8, [1])
        weight_zero_point = generate_input_initializer([1], np.int8, "weight_zero_point")
        Q_scale = onnx.helper.make_tensor_value_info("attn_output_scale", onnx.TensorProto.FLOAT, [1])
        attn_output_scale = generate_input_initializer([1], np.float32, "attn_output_scale")
        Q_zo = onnx.helper.make_tensor_value_info("attn_output_zero_point", onnx.TensorProto.INT8, [1])
        attn_output_zero_point = generate_input_initializer([1], np.int8, "attn_output_zero_point")
        Output = onnx.helper.make_tensor_value_info("attn_output_quantized", onnx.TensorProto.INT8, [13, 7])
        attention_node = onnx.helper.make_node(
            "QAttention",
            [
                "input_quantized",
                "weight_quantized",
                "bias",
                "input_scale",
                "weight_scale",
                "mask",
                "input_zero_point",
                "weight_zero_point",
            ],
            ["attn_output"],
            name="attention_quant",
        )
        qlinear_node = onnx.helper.make_node(
            "QuantizeLinear",
            ["attn_output", "attn_output_scale", "attn_output_zero_point"],
            ["attn_output_quantized"],
            name="attn_output_QuantizeLinear",
        )
        graph = onnx.helper.make_graph(
            [attention_node, qlinear_node],
            "test_graph_5",
            [
                Attention_input,
                Attention_weight,
                Attention_bias,
                Input_scale,
                Weight_scale,
                Attention_mask,
                Input_zo,
                Weight_zo,
                Q_scale,
                Q_zo,
            ],
            [Output],
        )
        graph.initializer.add().CopyFrom(weight_quantized)
        graph.initializer.add().CopyFrom(bias)
        graph.initializer.add().CopyFrom(input_scale)
        graph.initializer.add().CopyFrom(weight_scale)
        graph.initializer.add().CopyFrom(mask)
        graph.initializer.add().CopyFrom(input_zero_point)
        graph.initializer.add().CopyFrom(weight_zero_point)
        graph.initializer.add().CopyFrom(attn_output_scale)
        graph.initializer.add().CopyFrom(attn_output_zero_point)
        model = onnx.helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = calibrate.ONNXRTAugment(model, data_reader, [], white_nodes=["attention"])
        augment.augment_nodes = ["DequantizeLinear"]
        augment.already_quantized = True

        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = ["attention_quant", "attn_output_QuantizeLinear", "input_quantized_DequantizeLinear"]
        added_outputs = ["attn_output_quantized", "input_quantized_output", "attn_output"]
        self.assertEqual(len(augmented_model_node_names), 3)
        self.assertEqual(len(augmented_model_outputs), 3)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

        print("Finished TEST_CONFIG_5")

        #    QuantizeLinear
        #        |
        #    QLinearConv
        #        |
        #    DequantizeLinear
        A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        A_scale = onnx.helper.make_tensor_value_info("A_scale", onnx.TensorProto.FLOAT, [1])
        a_scale = generate_input_initializer([1], np.float32, "A_scale")
        A_zo = onnx.helper.make_tensor_value_info("A_zero_point", onnx.TensorProto.INT8, [1])
        a_zero_point = generate_input_initializer([1], np.int8, "A_zero_point")
        C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.INT8, [1, 1, 5, 5])
        c = generate_input_initializer([1, 1, 5, 5], np.int8, "C")
        C_scale = onnx.helper.make_tensor_value_info("C_scale", onnx.TensorProto.FLOAT, [1])
        c_scale = generate_input_initializer([1], np.float32, "C_scale")
        C_zo = onnx.helper.make_tensor_value_info("C_zero_point", onnx.TensorProto.INT8, [1])
        c_zero_point = generate_input_initializer([1], np.int8, "C_zero_point")
        E = onnx.helper.make_tensor_value_info("E", onnx.TensorProto.INT32, [1])
        e = generate_input_initializer([1], np.int32, "E")
        D_scale = onnx.helper.make_tensor_value_info("D_scale", onnx.TensorProto.FLOAT, [1])
        d_scale = generate_input_initializer([1], np.float32, "D_scale")
        D_zo = onnx.helper.make_tensor_value_info("D_zero_point", onnx.TensorProto.INT8, [1])
        d_zero_point = generate_input_initializer([1], np.int8, "D_zero_point")
        D = onnx.helper.make_tensor_value_info("D", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        quantize_node = onnx.helper.make_node(
            "QuantizeLinear", ["A", "A_scale", "A_zero_point"], ["A_quantized"], name="A_QuantizeLinear"
        )
        conv_node = onnx.helper.make_node(
            "QLinearConv",
            [
                "A_quantized",
                "A_scale",
                "A_zero_point",
                "C_quantized",
                "C_scale",
                "C_zero_point",
                "D_scale",
                "D_zero_point",
                "E",
            ],
            ["D_quantized"],
            name="conv_quant",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        dequantize_node = onnx.helper.make_node(
            "DequantizeLinear", ["D_quantized", "D_scale", "D_zero_point"], ["D"], name="D_DequantizeLinear"
        )
        graph = onnx.helper.make_graph(
            [quantize_node, conv_node, dequantize_node],
            "test_graph_5",
            [A, A_scale, A_zo, C, C_scale, C_zo, E, D_scale, D_zo],
            [D],
        )
        graph.initializer.add().CopyFrom(a_scale)
        graph.initializer.add().CopyFrom(a_zero_point)
        graph.initializer.add().CopyFrom(c)
        graph.initializer.add().CopyFrom(c_scale)
        graph.initializer.add().CopyFrom(c_zero_point)
        graph.initializer.add().CopyFrom(e)
        graph.initializer.add().CopyFrom(d_scale)
        graph.initializer.add().CopyFrom(d_zero_point)
        model = onnx.helper.make_model(graph)

        # Augmenting graph
        data_reader = None
        augment = calibrate.ONNXRTAugment(model, data_reader, [], white_nodes=["conv"])
        augment.augment_nodes = ["DequantizeLinear"]
        augment.already_quantized = True
        augment.augment_graph()
        augmented_model = augment.augmented_model

        augmented_model_node_names = [node.name for node in augmented_model.graph.node]
        augmented_model_outputs = [output.name for output in augmented_model.graph.output]
        added_node_names = [
            "A_QuantizeLinear",
            "conv_quant",
            "D_DequantizeLinear",
            "D_quantized_DequantizeLinear",
            "A_quantized_DequantizeLinear",
        ]
        added_outputs = ["D", "D_quantized_output", "A_quantized_output"]
        self.assertEqual(len(augmented_model_node_names), 5)
        self.assertEqual(len(augmented_model_outputs), 3)
        for name in added_node_names:
            self.assertTrue(name in augmented_model_node_names)
        for output in added_outputs:
            self.assertTrue(output in augmented_model_outputs)

    def test_quant_param_calculation(self):
        """TEST_CONFIG_6."""

        #   Relu
        #    |      \
        #   Conv     \
        #    |        \
        #   Relu       |
        #    |       Conv
        #   Conv      /
        #      \     /
        #         |
        #        Add

        input0 = onnx.helper.make_tensor_value_info("input0", onnx.TensorProto.FLOAT, [1, 3, 1, 3])
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 1, 3])

        X1_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X1_weight")
        X1_bias = generate_input_initializer([3], np.float32, "X1_bias")
        X3_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X3_weight")
        X3_bias = generate_input_initializer([3], np.float32, "X3_bias")
        X5_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X5_weight")
        X5_bias = generate_input_initializer([3], np.float32, "X5_bias")

        relu_node_1 = onnx.helper.make_node("Relu", ["input0"], ["X1"], name="Relu1")
        conv_node_1 = onnx.helper.make_node("Conv", ["X1", "X1_weight", "X1_bias"], ["X2"], name="Conv1")
        relu_node_2 = onnx.helper.make_node("Relu", ["X2"], ["X3"], name="Relu2")
        conv_node_2 = onnx.helper.make_node("Conv", ["X3", "X3_weight", "X3_bias"], ["X4"], name="Conv2")
        conv_node_3 = onnx.helper.make_node("Conv", ["X1", "X5_weight", "X5_bias"], ["X5"], name="Conv3")
        add_node = onnx.helper.make_node("Add", ["X4", "X5"], ["output"], name="Add")

        graph = onnx.helper.make_graph(
            [relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node],
            "test_graph_5",
            [input0],
            [output],
        )
        graph.initializer.add().CopyFrom(X1_weight)
        graph.initializer.add().CopyFrom(X1_bias)
        graph.initializer.add().CopyFrom(X3_weight)
        graph.initializer.add().CopyFrom(X3_bias)
        graph.initializer.add().CopyFrom(X5_weight)
        graph.initializer.add().CopyFrom(X5_bias)
        model = onnx.helper.make_model(graph, **{"opset_imports": [onnx.helper.make_opsetid("", 13)]})
        data_reader = DataReader()
        augment = calibrate.ONNXRTAugment(model, data_reader, ["Conv", "MatMul"])

        # test calculation of quantization params
        data_reader.rewind()
        quantization_params_dict = augment.dump_calibration({})
        data_reader.rewind()
        node_output_names, output_dicts_list = augment.get_intermediate_outputs({})
        data_reader.rewind()
        dict_for_quantization = augment._map_calibration(node_output_names, output_dicts_list)
        # check the size of the quantization dictionary

        self.assertEqual(len(quantization_params_dict), 12)

        # check the computation of zp and scale
        for key, value in quantization_params_dict.items():
            self.assertTrue(value is not None)
            self.assertTrue(len(value) == 2)

            thresholds = dict_for_quantization[key]
            rmin = min(thresholds[0], 0)
            rmax = max(thresholds[1], 0)
            if key == "X2":  # next_node is Relu
                if rmin < 0:
                    rmin = 0

            scale_expected = np.float32((rmax - rmin) / 255 if rmin != rmax else 1)
            zp_expected = np.uint8(round(max(0, min(255, (0 - rmin) / scale_expected))))
            zp_actual = value[0]
            scale_actual = value[1]

            self.assertAlmostEqual(zp_expected, zp_actual)
            self.assertAlmostEqual(scale_expected, scale_actual)

        print("Finished" + " test calculation of quantization params.")

    def test_calibrator(self):
        regular_data = [np.arange(15).reshape(3, 5).astype("float32"), np.arange(15).reshape(3, 5).astype("float32")]
        irregular_data = [np.arange(10).reshape(2, 5).astype("float32"), np.arange(5).reshape(1, 5).astype("float32")]

        calib = calibrator.CALIBRATOR["MinMax"]()
        calib.collect(irregular_data)
        res = calib.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(9.0).astype(np.float32))
        calib.collect(regular_data)
        res = calib.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(14.0).astype(np.float32))
        calib.clear()
        res = calib.calib_range
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])
        del calib

        calib = calibrator.CALIBRATOR["Entropy"]()
        calib.collect(irregular_data)
        res = calib.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(9.0).astype(np.float32))
        calib.collect(regular_data)
        res = calib.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(9.140625).astype(np.float32))
        calib.clear()
        res = calib.calib_range
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])
        del calib

        calib = calibrator.CALIBRATOR["Percentile"]()
        calib.collect(irregular_data)
        res = calib.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(8.991211).astype(np.float32))
        calib.collect(regular_data)
        res = calib.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(13.9921875).astype(np.float32))
        calib.clear()
        res = calib.calib_range
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])
        del calib


if __name__ == "__main__":
    unittest.main()
