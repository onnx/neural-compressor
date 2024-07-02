import re
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--build_id", required=True, type=str)
args = parser.parse_args()

URL = (
    "https://dev.azure.com/lpot-inc/onnx-neural-compressor/_build/results?buildId="
    + args.build_id
    + "&view=artifacts&pathAsName=false&type=publishedArtifacts"
)


def str_to_float(value):
    try:
        return round(float(value), 4)
    except ValueError:
        return value


def main():
    result_dict = {
        args.model: {
            "performance": {"value": "N/A", "log_path": URL},
            "accuracy": {"value": "N/A", "log_path": URL},
        }
    }

    pattern = {
        "performance": r"Throughput: ([\d.]+)",
        "accuracy": r"Accuracy: ([\d.]+)",
    }

    for mode, _ in result_dict[args.model].items():
        log_file = f"/neural-compressor/.azure-pipelines/scripts/models/{args.model}/{mode}.log"
        if not os.path.exists(log_file):
            print(f"The file '{log_file}' does not exist.")
            continue

        with open(log_file, "r") as file:
            log_content = file.read()

        match = re.search(pattern[mode], log_content)

        if match:
            result_dict[args.model][mode]["value"] = str_to_float(match.group(1))

    with open(f"/neural-compressor/.azure-pipelines/scripts/models/{args.model}/result.json", "w") as json_file:
        json.dump(result_dict, json_file, indent=4)


if __name__ == "__main__":
    main()
