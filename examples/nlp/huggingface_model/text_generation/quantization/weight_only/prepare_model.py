import argparse
import os
import subprocess

import optimum.version
from packaging import version

OPTIMUM114_VERSION = version.Version("1.14.0")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=True, default="")
    parser.add_argument("--output_model", type=str, required=False, default=None)
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        default="text-generation-with-past",
        choices=["text-generation-with-past", "text-generation"],
    )
    args = parser.parse_args()
    if args.output_model is None:
        args.output_model = os.path.basename(args.input_model) + "-onnx"
    return args


def prepare_model(input_model, output_model, task):
    print("\nexport model...")
    if version.Version(optimum.version.__version__) < OPTIMUM114_VERSION:
        raise ImportError("Please upgrade optimum to >= 1.14.0")

    subprocess.run(
        [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            f"{input_model}",
            "--task",
            task,
            f"{output_model}",
            "--trust-remote-code",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    assert os.path.exists(output_model), f"{output_model} doesn't exist!"


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model, args.task)
