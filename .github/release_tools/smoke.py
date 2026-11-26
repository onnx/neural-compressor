# Copyright (c) ONNX Neural Compressor Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify installed metadata and public quantization APIs outside the checkout."""

import argparse
import importlib.metadata
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

import onnx_neural_compressor as onc
from onnx_neural_compressor import data_reader, version
from onnx_neural_compressor.quantization import QuantFormat, config, quantize


class Reader(data_reader.CalibrationDataReader):
    def __init__(self, samples):
        self.samples = samples
        self.rewind()

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter(self.samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_version")
    args = parser.parse_args()
    installed = importlib.metadata.version("onnx-neural-compressor")
    assert onc.__version__ == version.__version__ == installed == args.expected_version
    print(f"Installed {installed} from {onc.__file__}")
    rng = np.random.default_rng(7)
    weights = rng.standard_normal((16, 8)).astype(np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["input", "weight"], ["output"], name="matmul")],
            "release_smoke",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 16])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])],
            [onnx.numpy_helper.from_array(weights, name="weight")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 17)],
        ir_version=8,
    )
    samples = [{"input": rng.standard_normal((1, 16)).astype(np.float32)} for _ in range(8)]
    configs = {
        "dynamic": config.DynamicQuantConfig(),
        "static": config.StaticQuantConfig(Reader(samples), quant_format=QuantFormat.QDQ),
    }
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "model.onnx"
        onnx.save(model, source)
        for name, quant_config in configs.items():
            output = Path(directory) / f"{name}.onnx"
            quantize(str(source), str(output), quant_config)
            onnx.checker.check_model(str(output))
            quantized = onnx.load(output)
            assert quantized.producer_version == args.expected_version
            operators = {node.op_type for node in quantized.graph.node}
            assert operators & {"MatMulInteger", "QuantizeLinear"}, operators
            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            actual = session.run(None, samples[0])[0]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, samples[0]["input"] @ weights, rtol=0.1, atol=0.15)
    print("Dynamic and static quantization round trips passed")


if __name__ == "__main__":
    main()
