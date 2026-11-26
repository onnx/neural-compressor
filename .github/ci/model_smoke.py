"""Offline CPU round trips for small, seeded BERT and ResNet architectures.

These check model conversion and inference, not pretrained-model accuracy or
hardware performance. All weights and calibration inputs are generated locally.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import BertConfig, BertForSequenceClassification, ResNetConfig, ResNetForImageClassification

from onnx_neural_compressor import data_reader
from onnx_neural_compressor.quantization import QuantFormat, config, quantize


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model(*inputs).logits


class Reader(data_reader.CalibrationDataReader):
    def __init__(self, batches):
        self.batches = batches
        self.rewind()

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter(self.batches)


def make_model(name):
    torch.manual_seed(7)
    if name == "bert":
        model = BertForSequenceClassification(
            BertConfig(
                vocab_size=128,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=64,
                max_position_embeddings=32,
                num_labels=4,
                attn_implementation="eager",
            )
        )
        names = ["input_ids", "attention_mask"]
        inputs = (torch.randint(0, 128, (1, 16)), torch.ones((1, 16), dtype=torch.int64))
    else:
        model = ResNetForImageClassification(
            ResNetConfig(
                embedding_size=8,
                hidden_sizes=[16, 32, 64, 128],
                depths=[1, 1, 1, 1],
                layer_type="basic",
                num_labels=4,
            )
        )
        names = ["pixel_values"]
        inputs = (torch.randn(1, 3, 32, 32),)
    # Default zero biases are aliased by export/optimization. Use distinct seeded
    # biases, as in trained models, so the fixture exercises ordinary model graphs.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.normal_(mean=0.0, std=0.01)
    return Logits(model).eval(), names, inputs


def run(model_name, mode, directory):
    torch.set_num_threads(1)
    model, names, inputs = make_model(model_name)
    original = directory / "original.onnx"
    converted = directory / "quantized.onnx"
    torch.onnx.export(model, inputs, str(original), input_names=names, output_names=["logits"], opset_version=17)
    rng = np.random.default_rng(7)

    def batch():
        if model_name == "bert":
            return {
                "input_ids": rng.integers(0, 128, size=(1, 16), dtype=np.int64),
                "attention_mask": np.ones((1, 16), dtype=np.int64),
            }
        return {"pixel_values": rng.standard_normal((1, 3, 32, 32)).astype(np.float32)}

    if mode == "dynamic":
        quant_config = config.DynamicQuantConfig(op_types_to_quantize=["MatMul"])
    else:
        quant_config = config.StaticQuantConfig(
            calibration_data_reader=Reader([batch() for _ in range(8)]),
            quant_format=QuantFormat.QDQ if mode == "qdq" else QuantFormat.QOperator,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        )
    quantize(str(original), str(converted), quant_config)
    onnx.checker.check_model(str(converted))
    saved_model = onnx.load(str(converted))
    expected_ops = {
        "dynamic": {"MatMulInteger", "DynamicQuantizeLinear"},
        "qdq": {"QuantizeLinear", "DequantizeLinear"},
        "qoperator": {"QLinearConv", "QLinearMatMul", "QGemm"},
    }[mode]
    quantized_nodes = sum(node.op_type in expected_ops for node in saved_model.graph.node)
    if not quantized_nodes:
        raise AssertionError(f"No quantized operators found for {model_name}/{mode}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        for path in (original, converted)
    ]
    errors = []
    for _ in range(3):
        sample = batch()
        reference, actual = [session.run(None, sample)[0] for session in sessions]
        assert actual.shape == reference.shape == (1, 4)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, reference, rtol=0.2, atol=0.02)
        errors.append(float(np.max(np.abs(actual - reference))))
    return {
        "model": model_name,
        "mode": mode,
        "provider": "CPUExecutionProvider",
        "quantized_nodes": quantized_nodes,
        "max_absolute_error": max(errors),
        "inference_samples": len(errors),
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["bert", "resnet"], required=True)
    parser.add_argument("--mode", choices=["dynamic", "qdq", "qoperator"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "mode": args.mode, "status": "failed"}
    try:
        with tempfile.TemporaryDirectory() as temp:
            report.update(run(args.model, args.mode, Path(temp)))
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
