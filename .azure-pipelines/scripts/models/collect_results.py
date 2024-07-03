import argparse
import json
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--build_id", required=True, type=str)
args = parser.parse_args()

URL = (
    "https://dev.azure.com/lpot-inc/onnx-neural-compressor/_build/results?buildId="
    + args.build_id
    + "&view=artifacts&pathAsName=false&type=publishedArtifacts"
)
REFER_SUMMARY_PATH = "/neural-compressor/.azure-pipelines/scripts/models/refer_summary.json"


def str_to_float(value):
    try:
        return round(float(value), 4)
    except ValueError:
        return value


def get_refer_data():
    if not os.path.exists(REFER_SUMMARY_PATH):
        print(f"The file '{REFER_SUMMARY_PATH}' does not exist.")
        return {}

    with open(REFER_SUMMARY_PATH, "r") as file:
        refer = json.load(file)
    return refer


def check_status(performance, accuracy):
    refer = get_refer_data()

    refer_accuracy = refer.get(args.model, {}).get("accuracy", {}).get("value", "N/A")
    refer_performance = refer.get(args.model, {}).get("performance", {}).get("value", "N/A")

    assert accuracy != "N/A" and performance != "N/A"
    if refer_accuracy != "N/A":
        assert abs(accuracy - refer_accuracy) <= 0.001
    if refer_performance != "N/A":
        assert (refer_performance - performance) / refer_performance <= 0.08


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

    check_status(result_dict[args.model]["performance"]["value"], result_dict[args.model]["accuracy"]["value"])


if __name__ == "__main__":
    main()
