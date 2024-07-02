import re
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--build_id", required=True, type=str)
args = parser.parse_args()

result_dict = {
    args.model: {
        "performance": {"value": "n/a", "log_path": "n/a"},
        "accuracy": {"value": "n/a", "log_path": "n/a"},
    }
}

pattern = {
    "performance": r"Throughput = ([\d.]+)",
    "accuracy": r"Accuracy: ([\d.]+)",
}

for mode, info in result_dict[args.model].items():
    log_file = f"/neural-compressor/.azure-pipelines/scripts/models/{args.model}/{mode}.log"
    if not os.path.exists(log_file):
        print(f"The file '{log_file}' does not exist.")
        continue

    with open(log_file, "r") as file:
        log_content = file.read()

    match = re.search(pattern[mode], log_content)

    if match:
        result_dict[args.model][mode]["value"] = match.group(1)


with open(f"/neural-compressor/.azure-pipelines/scripts/models/{args.model}/result.json", "w") as json_file:
    json.dump(result_dict, json_file, indent=4)
