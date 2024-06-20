import copy
import os
import shutil
import unittest

import numpy as np
import onnx
import torch
import torch.nn as nn
import transformers
from onnxruntime.transformers import fusion_options, optimizer
from optimum.exporters.onnx import main_export

from onnx_neural_compressor import config, logger, onnx_model


def find_onnx_file(folder_path):
    # return first .onnx file path in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None


class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


def build_matmul_model():
    # MatMul - Add - Add
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 5, 5])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1, 5, 2])
    D = onnx.helper.make_tensor_value_info("D", onnx.TensorProto.FLOAT, [1, 5, 2])
    H = onnx.helper.make_tensor_value_info("H", onnx.TensorProto.FLOAT, [1, 5, 2])

    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    B_init = onnx.helper.make_tensor("B", onnx.TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    E_init = onnx.helper.make_tensor("E", onnx.TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())

    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    add = onnx.helper.make_node("Add", ["C", "E"], ["D"], name="add")

    f_value = np.random.randint(2, size=(10)).astype(np.float32)
    # F_init = onnx.helper.make_tensor("F", onnx.TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())
    add2 = onnx.helper.make_node("Add", ["D", "E"], ["H"], name="add2")

    graph = onnx.helper.make_graph([matmul_node, add, add2], "test_graph_1", [A], [H], [B_init, E_init])
    model = onnx.helper.make_model(graph)
    model = onnx.helper.make_model(graph, **{"opset_imports": [onnx.helper.make_opsetid("", 13)]})
    return model


class TestONNXModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="gptj",
        )
        self.gptj = find_onnx_file("./gptj")

        model = Net(512, 1024 * 1024)
        input = torch.randn(512, requires_grad=True)
        folder_path = "./large_model"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with torch.no_grad():
            torch.onnx.export(
                model, (input,), os.path.join(folder_path, "model.onnx"), do_constant_folding=True, opset_version=13
            )
        self.large_model = os.path.join(folder_path, "model.onnx")

        matmul_add_model = build_matmul_model()
        onnx.save(matmul_add_model, "matmul_add.onnx")
        self.matmul_add_model = "matmul_add.onnx"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./gptj", ignore_errors=True)
        shutil.rmtree("./large_model", ignore_errors=True)
        os.remove("matmul_add.onnx")

    def setUp(self):
        # print the test name
        logger.info(f"Running ONNXRT TestONNXModel test: {self.id()}")

    def test_model_atrribute(self):
        model = onnx_model.ONNXModel(self.gptj)
        # hf_config
        self.assertTrue(isinstance(model.hf_config, transformers.PretrainedConfig))

        model = onnx_model.ONNXModel(self.matmul_add_model)
        # model_path
        self.assertEqual(model.model_path, self.matmul_add_model)
        # framework
        self.assertEqual(model.framework(), "onnxruntime")
        # q_config
        quant_config = config.RTNConfig()
        model.q_config = quant_config
        self.assertTrue(isinstance(model.q_config, config.RTNConfig))
        # input
        self.assertEqual(len(model.input()), 1)
        # output
        self.assertEqual(len(model.output()), 1)
        # graph_info
        self.assertEqual(len(model.graph_info), 3)

    def test_check_is_large_model(self):
        # model <= 2GB
        model = onnx_model.ONNXModel(self.gptj)  # pass str
        model.check_is_large_model()
        self.assertFalse(model.is_large_model)

        model = onnx.load(self.gptj)
        model = onnx_model.ONNXModel(model)  # pass ModelProto
        model.check_is_large_model()
        self.assertFalse(model.is_large_model)

        # model > 2GB
        model = onnx_model.ONNXModel(self.large_model)  # pass str
        model.check_is_large_model()
        self.assertTrue(model.is_large_model)

        model = onnx.load(self.large_model)
        model = onnx_model.ONNXModel(model)  # pass ModelProto
        model.check_is_large_model()
        self.assertTrue(model.is_large_model)

    def test_save(self):
        # test save
        model = onnx_model.ONNXModel(self.gptj)
        save_path = "./gptj/test.onnx"
        model.save(save_path)

        # test large model save
        model = onnx_model.ONNXModel(self.large_model)
        save_path = ".large_model_save.onnx"
        model.save(save_path)

        # test save path does not exist
        with self.assertRaises(ValueError) as cm:
            save_path = "./gptj_output/test.onnx"
            model.save(save_path)
        self.assertEqual(str(cm.exception), '"root" directory does not exists.')

    def test_get_initializer_share_num(self):
        model = onnx_model.ONNXModel(self.matmul_add_model)
        init_num = model.get_initializer_share_num("E")
        self.assertEqual(init_num, 2)
        init_num = model.get_initializer_share_num("B")
        self.assertEqual(init_num, 1)

    def test_get_node(self):
        model = onnx_model.ONNXModel(self.matmul_add_model)
        node = model.get_node("Matmul")
        self.assertEqual(node.name, "Matmul")
        node = model.get_node("test")
        self.assertTrue(node is None)

    def test_get_node_by_weight(self):
        model = onnx_model.ONNXModel(self.matmul_add_model)
        model._input_name_to_nodes = {}
        node = model.get_node_by_weight("B")
        self.assertEqual(node.name, "Matmul")

    def test_set_initializer(self):
        model = onnx_model.ONNXModel(self.matmul_add_model)
        array = np.ones((5, 2))
        model.set_initializer("B", array)
        new_init_array = onnx.numpy_helper.to_array(model.get_initializer("B"))
        self.assertTrue((new_init_array == array).all())

    def test_get_siblings(self):
        model = onnx_model.ONNXModel(self.gptj)
        node = model.get_node("/h.0/attn/q_proj/MatMul")
        siblings = model.get_siblings(node)
        self.assertEqual(len(siblings), 3)

    def test_replace_input_of_all_nodes(self):
        model = onnx_model.ONNXModel(self.matmul_add_model)
        model.replace_input_of_all_nodes("C", "new_C")
        self.assertTrue("new_C" in model.get_node("add").input)

        model.replace_input_of_all_nodes("C", "new_C", white_optype=["Add"])
        self.assertTrue("new_C" in model.get_node("add").input)

    def test_replace_output_of_all_nodes(self):
        model = onnx_model.ONNXModel(self.matmul_add_model)
        model.replace_output_of_all_nodes("D", "new_D")
        self.assertTrue("new_D" in model.get_node("add").output)

        model.replace_output_of_all_nodes("D", "new_D", white_optype=["Add"])
        self.assertTrue("new_D" in model.get_node("add").output)

    def test_remove_unused_nodes(self):
        # test unused Constant node
        input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [3, 1])
        values = np.random.randn(5, 5).astype(np.float32)
        constant_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["constant_output"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )
        matmul1_weight = onnx.helper.make_tensor(
            "matmul1_weight", onnx.TensorProto.FLOAT, [1, 3], np.random.random((1, 3)).reshape(3).tolist()
        )
        matmul1_output = onnx.helper.make_tensor_value_info("matmul1_output", onnx.TensorProto.FLOAT, [3, 3])
        matmul1_node = onnx.helper.make_node("MatMul", ["input", "matmul1_weight"], ["matmul1_output"], name="Matmul_0")

        initializers = [matmul1_weight]
        graph = onnx.helper.make_graph(
            [matmul1_node],
            "test_model",
            [input_tensor],
            [matmul1_output],
            initializer=initializers,
        )
        graph.node.append(constant_node)
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
        model.ir_version = 7
        model = onnx_model.ONNXModel(model)
        self.assertTrue("Constant" in [node.op_type for node in model.nodes()])
        model._input_name_to_nodes, model._output_name_to_node = (
            {},
            {},
        )  # test if _input_name_to_nodes and _output_name_to_node is empty
        model.remove_unused_nodes()
        self.assertTrue("Constant" not in [node.op_type for node in model.nodes()])

        # test unused QuantizeLinear and DequantizeLinear
        A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        A_scale = onnx.helper.make_tensor_value_info("A_scale", onnx.TensorProto.FLOAT, [1])
        a_scale = onnx.numpy_helper.from_array(np.random.ranf([1]).astype(np.float32), "A_scale")
        A_zero = onnx.helper.make_tensor_value_info("A_zero_point", onnx.TensorProto.INT8, [1])
        a_zero_point = onnx.numpy_helper.from_array(np.random.ranf([1]).astype(np.int8), "A_zero_point")
        quantize_node = onnx.helper.make_node(
            "QuantizeLinear", ["A", "A_scale", "A_zero_point"], ["B_quantized"], name="quantizelinear"
        )

        B_scale = onnx.helper.make_tensor_value_info("B_scale", onnx.TensorProto.FLOAT, [1])
        b_scale = onnx.numpy_helper.from_array(np.random.ranf([1]).astype(np.float32), "B_scale")
        B_zero = onnx.helper.make_tensor_value_info("B_zero_point", onnx.TensorProto.INT8, [1])
        b_zero_point = onnx.numpy_helper.from_array(np.random.ranf([1]).astype(np.int8), "B_zero_point")
        C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
        dequantize_node = onnx.helper.make_node(
            "DequantizeLinear", ["B_quantized", "B_scale", "B_zero_point"], ["C"], name="dequantizelinear"
        )

        graph = onnx.helper.make_graph(
            [quantize_node, dequantize_node],
            "test_model",
            [A],
            [C],
            initializer=[a_scale, a_zero_point, b_scale, b_zero_point],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
        model.ir_version = 7
        model = onnx_model.ONNXModel(model)
        self.assertTrue("QuantizeLinear" in [node.op_type for node in model.nodes()])
        self.assertTrue("DequantizeLinear" in [node.op_type for node in model.nodes()])
        model.remove_unused_nodes()
        self.assertTrue("QuantizeLinear" not in [node.op_type for node in model.nodes()])
        self.assertTrue("DequantizeLinear" not in [node.op_type for node in model.nodes()])

        # test node which does not serve as the input or output of any other nodes
        matmul2_weight = onnx.helper.make_tensor(
            "matmul2_weight", onnx.TensorProto.FLOAT, [1, 3], np.random.random((1, 3)).reshape(3).tolist()
        )
        matmul2_output = onnx.helper.make_tensor_value_info("matmul2_output", onnx.TensorProto.FLOAT, [3, 3])
        matmul2_node = onnx.helper.make_node("MatMul", ["input", "matmul2_weight"], ["matmul2_output"], name="Matmul_1")
        graph.node.append(matmul2_node)
        graph.initializer.append(matmul2_weight)
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
        model.ir_version = 7
        model = onnx_model.ONNXModel(model)
        self.assertTrue("MatMul" in [node.op_type for node in model.nodes()])
        model.remove_unused_nodes()
        self.assertTrue("MatMul" not in [node.op_type for node in model.nodes()])

    def test_add_or_remove_tensors_to_outputs(self):
        model = onnx_model.ONNXModel(self.matmul_add_model)
        model.add_tensors_to_outputs(["test_out1", "test_out2"])
        self.assertTrue("test_out1" in [out.name for out in model.model.graph.output])
        self.assertTrue("test_out2" in [out.name for out in model.model.graph.output])

        model.remove_tensors_from_outputs(["test_out1", "test_out2"])
        self.assertTrue("test_out1" not in [out.name for out in model.model.graph.output])
        self.assertTrue("test_out2" not in [out.name for out in model.model.graph.output])

    def test_re_org_output(self):
        model = onnx_model.ONNXModel(self.gptj)
        origin_output = copy.deepcopy(model.output())
        first_output = model.model.graph.output[0]
        model.model.graph.output.remove(first_output)
        model.model.graph.output.append(first_output)
        self.assertNotEqual(model.output()[0], origin_output[0])
        model.re_org_output(origin_output)
        self.assertEqual(model.output()[0], origin_output[0])


if __name__ == "__main__":
    unittest.main()
