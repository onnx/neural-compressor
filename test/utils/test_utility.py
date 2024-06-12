"""Tests for utility components."""
import os
import unittest
import shutil
import onnx
import numpy as np
import optimum.exporters.onnx
import onnxruntime
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer

from onnx_neural_compressor import utility
from onnx_neural_compressor import onnx_model

def find_onnx_file(folder_path):
    # return first .onnx file path in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".onnx"):
                return os.path.join(root, file)
    return None

class TestOptions(unittest.TestCase):

    def test_set_random_seed(self):
        seed = 12345
        utility.set_random_seed(seed)
        self.assertEqual(utility.options.random_seed, seed)

        # non int type
        seed = "12345"
        with self.assertRaises(AssertionError):
            utility.set_random_seed(seed)

    def test_set_workspace(self):
        workspace = "/path/to/workspace"
        utility.set_workspace(workspace)
        self.assertEqual(utility.options.workspace, workspace)

        # non String type
        workspace = 12345
        with self.assertRaises(AssertionError):
            utility.set_workspace(workspace)

    def test_set_resume_from(self):
        resume_from = "/path/to/resume"
        utility.set_resume_from(resume_from)
        self.assertEqual(utility.options.resume_from, resume_from)

        # non String type
        resume_from = 12345
        with self.assertRaises(AssertionError):
            utility.set_resume_from(resume_from)


class TestCPUInfo(unittest.TestCase):

    def test_cpu_info(self):
        cpu_info = utility.CpuInfo()
        assert cpu_info.cores_per_socket > 0, "CPU count should be greater than 0"
        assert isinstance(cpu_info.bf16, bool), "bf16 should be a boolean"
        assert isinstance(cpu_info.vnni, bool), "avx512 should be a boolean"


class TestLazyImport(unittest.TestCase):

    def test_lazy_import(self):
        # Test import
        pydantic = utility.LazyImport("pydantic")
        assert pydantic.__name__ == "pydantic", "pydantic should be imported"

    def test_lazy_import_error(self):
        # Test import error
        with self.assertRaises(ImportError):
            non_existent_module = utility.LazyImport("non_existent_module")
            non_existent_module.non_existent_function()


class TestSingletonDecorator:

    def test_singleton_decorator(self):

        @utility.singleton
        class TestSingleton:

            def __init__(self):
                self.value = 0

        instance = TestSingleton()
        instance.value = 1
        instance2 = TestSingleton()
        assert instance2.value == 1, "Singleton should return the same instance"

class TestGetVersion(unittest.TestCase):

    def test_get_version(self):
        from onnx_neural_compressor import version

        self.assertTrue(isinstance(version.__version__, True))

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

    def test_check_value(self):
        src = [1, 2, 3]
        supported_type = int
        supported_value = [1, 2, 3]
        result = utility.check_value("name", src, supported_type, supported_value)
        self.assertTrue(result)

        src = [1, 2, 3]
        supported_type = list
        with self.assertRaises(AssertionError) as cm:
            utility.check_value("name", src, supported_type)
        self.assertEqual(
            str(cm.exception),
            "Type of 'name' items should be <class 'list'> but not [<class 'int'>, <class 'int'>, <class 'int'>]")

        src = 1
        supported_type = list
        with self.assertRaises(AssertionError) as cm:
            utility.check_value("name", src, supported_type)
        self.assertEqual(
            str(cm.exception),
            "Type of 'name' should be <class 'list'> but not <class 'int'>")

        src = "a"
        supported_type = str
        supported_value = ["b"]
        with self.assertRaises(AssertionError) as cm:
            utility.check_value("name", src, supported_type, supported_value)
        self.assertEqual(
            str(cm.exception),
            "'a' is not in supported 'name': ['b']. Skip setting it.")

        src = ["a"]
        supported_type = str
        supported_value = ["b"]
        with self.assertRaises(AssertionError) as cm:
            utility.check_value("name", src, supported_type, supported_value)
        self.assertEqual(
            str(cm.exception),
            "['a'] is not in supported 'name': ['b']. Skip setting it.")

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
        self.assertTrue(utility.is_B_transposed(node))

        node = onnx.helper.make_node(
            "Gemm",
            inputs=["a", "b", "c"],
            outputs=["y"],
            alpha=0.25,
            beta=0.35,
        )
        self.assertFalse(utility.is_B_transposed(node))

    def test_get_qrange_for_qType(self):
        qrange = utility.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.UINT8)
        self.assertEqual(qrange, 255)
        qrange = utility.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.UINT8, reduce_range=True)
        self.assertEqual(qrange, 127)
        qrange = utility.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.INT8)
        self.assertEqual(qrange, 254)
        qrange = utility.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.INT8, reduce_range=True)
        self.assertEqual(qrange, 128)

        # unexpected quantization data type
        with self.assertRaises(ValueError) as cm:
            utility.get_qrange_for_qType(qType=onnx.onnx_pb.TensorProto.FLOAT)
        self.assertEqual(str(cm.exception), "unsupported quantization data type")

    def test_quantize_data(self):
        # sym int8
        data = [1, 2, 3, 4, 5]
        quantize_range = 127
        qType = onnx.onnx_pb.TensorProto.INT8
        scheme = "sym"
        rmin, rmax, zero_point, scale, quantized_data = utility.quantize_data(data, quantize_range, qType, scheme)
        self.assertEqual(quantized_data.dtype, np.int8)

        scale, zero_point = utility._calculate_scale_zp(
            np.array([0]), np.array([5]), quantize_range, qType, scheme)
        self.assertEqual(zero_point.dtype, np.int8)

        scale, zero_point = utility._calculate_scale_zp(
            np.array([0]), np.array([127]), quantize_range, qType, scheme)
        self.assertEqual(zero_point.dtype, np.int8)

        # asym uint8
        data = [-1, 0, 1, 2, 3]
        quantize_range = 255
        qType = onnx.onnx_pb.TensorProto.UINT8
        scheme = "asym"
        rmin, rmax, zero_point, scale, quantized_data = utility.quantize_data(data, quantize_range, qType, scheme)
        self.assertEqual(quantized_data.dtype, np.uint8)

        scale, zero_point = utility._calculate_scale_zp(
            np.array([0]), np.array([5]), quantize_range, qType, scheme)
        self.assertEqual(zero_point.dtype, np.uint8)

        scale, zero_point = utility._calculate_scale_zp(
            np.array([0]), np.array([255]), quantize_range, qType, scheme)
        self.assertEqual(zero_point.dtype, np.uint8)

        # unexpected combination
        with self.assertRaises(ValueError) as cm:
            rmin, rmax, zero_point, scale, quantized_data = utility.quantize_data(
                data, quantize_range, qType=onnx.onnx_pb.TensorProto.UINT8, scheme="sym")
        self.assertTrue("Unexpected combination of data type" in str(cm.exception))


    def test_check_model_with_infer_shapes(self):
        self.assertFalse(utility.check_model_with_infer_shapes(self.llama))
        self.assertTrue(utility.check_model_with_infer_shapes(self.llama_optimized))
        self.assertTrue(utility.check_model_with_infer_shapes(
            onnx_model.ONNXModel(onnx.load(self.llama_optimized, load_external_data=False))
        ))

if __name__ == "__main__":
    unittest.main()
