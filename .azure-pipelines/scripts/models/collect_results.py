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
        "performance": {"reg": r"Throughput = ([\d.]+)", "value": "n/a", "log_path": "n/a"},
        "accuracy": {"reg": r"Accuracy: ([\d.]+)", "value": "n/a", "log_path": "n/a"},
    }
}

for key, info in result_dict[args.model].items():
    log_file = f"/neural-compressor/.azure-pipelines/scripts/models/{args.model}/{key}.log"
    if not os.path.exists(log_file):
        print(f"The file '{log_file}' does not exist.")
        continue

    with open(log_file, "r") as file:
        log_content = file.read()

    match = re.search(info["reg"], log_content)

    if match:
        result_dict[args.model][key]["value"] = match.group(1)


with open(f"/neural-compressor/.azure-pipelines/scripts/models/{args.model}/result.json", "w") as json_file:
    json.dump(result_dict, json_file, indent=4)
