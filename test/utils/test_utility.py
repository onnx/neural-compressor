"""Tests for utility components."""

import os
import shutil
import unittest

import onnx
import onnxruntime
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
import optimum.exporters.onnx

from onnx_neural_compressor import onnx_model, utility


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

        self.assertTrue(isinstance(version.__version__, str))


class TestUtilityFunctions(unittest.TestCase):

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
            "Type of 'name' items should be <class 'list'> but not [<class 'int'>, <class 'int'>, <class 'int'>]",
        )

        src = 1
        supported_type = list
        with self.assertRaises(AssertionError) as cm:
            utility.check_value("name", src, supported_type)
        self.assertEqual(str(cm.exception), "Type of 'name' should be <class 'list'> but not <class 'int'>")

        src = "a"
        supported_type = str
        supported_value = ["b"]
        with self.assertRaises(AssertionError) as cm:
            utility.check_value("name", src, supported_type, supported_value)
        self.assertEqual(str(cm.exception), "'a' is not in supported 'name': ['b']. Skip setting it.")

        src = ["a"]
        supported_type = str
        supported_value = ["b"]
        with self.assertRaises(AssertionError) as cm:
            utility.check_value("name", src, supported_type, supported_value)
        self.assertEqual(str(cm.exception), "['a'] is not in supported 'name': ['b']. Skip setting it.")


if __name__ == "__main__":
    unittest.main()
