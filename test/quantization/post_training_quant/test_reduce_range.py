# Copyright (c) ONNX Neural Compressor Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Public quantization configurations must control the stored weight range."""

import numpy as np
import onnx
import pytest

from onnx_neural_compressor.data_reader import CalibrationDataReader
from onnx_neural_compressor.quantization import QuantFormat, config, quantize


class Reader(CalibrationDataReader):
    def get_next(self):
        return next(self.samples, None)

    def rewind(self):
        self.samples = iter([{"input": np.ones((1, 16), dtype=np.float32)}])


@pytest.mark.parametrize("mode", ["dynamic", "qdq", "qoperator"])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("reduce_range", [False, True])
def test_stored_weight_range(tmp_path, mode, per_channel, reduce_range):
    weights = np.random.default_rng(7).standard_normal((16, 8)).astype(np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["input", "weight"], ["output"], name="matmul")],
            "weight_range",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 16])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 8])],
            [onnx.numpy_helper.from_array(weights, name="weight")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 17)],
        ir_version=8,
    )
    source, output = tmp_path / "model.onnx", tmp_path / "quantized.onnx"
    onnx.save(model, source)
    options = dict(per_channel=per_channel, reduce_range=reduce_range)
    if mode == "dynamic":
        quant_config = config.DynamicQuantConfig(**options)
    else:
        quant_config = config.StaticQuantConfig(
            Reader(), quant_format=QuantFormat.QDQ if mode == "qdq" else QuantFormat.QOperator, **options
        )
    quantize(str(source), str(output), quant_config)
    result = onnx.load(output)
    onnx.checker.check_model(result)
    quantized_weights = next(tensor for tensor in result.graph.initializer if tensor.name == "weight_quantized")
    assert quantized_weights.data_type == onnx.TensorProto.INT8
    values = onnx.numpy_helper.to_array(quantized_weights).astype(np.int32)
    if reduce_range:
        assert values.min() >= -64
        assert values.max() <= 64
    else:
        assert np.abs(values).max() > 64
