import argparse
import json
import os

from jinja2 import Environment, FileSystemLoader

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--json_path", type=str, required=True)
parser.add_argument("--last_json_path", type=str, required=True)
args = parser.parse_args()


def get_data(json_path):
    """
    {
        model: {
            "performance": {"value": "N/A"|number, "log_path": string},
            "accuracy": {"value": "N/A"|number, "log_path": string},
        }
    }
    """
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        return {}


def get_ratio(cur, last):
    if cur == "N/A" or last == "N/A":
        ratio = "N/A"
    else:
        ratio = (float(cur) - float(last)) / float(last) * 100
        ratio = round(float(ratio), 2)
        ratio = f"{ratio}%"
    return ratio


def get_accuracy_ratio(current_json, last_accuracy_dict):
    compare_result_dict = []
    for model, item in current_json.items():
        current_accuracy = item.get("accuracy", {}).get("value", "N/A")
        last_accuracy = last_accuracy_dict.get(model, {}).get("accuracy", {}).get("value", "N/A")
        accuracy_ratio = get_ratio(current_accuracy, last_accuracy)

        current_performance = item.get("performance", {}).get("value", "N/A")
        last_performance = last_accuracy_dict.get(model, {}).get("performance", {}).get("value", "N/A")
        performance_ratio = get_ratio(current_performance, last_performance)

        if current_accuracy == "N/A" or current_performance == "N/A":
            status = "FAILURE"
        elif accuracy_ratio != 0:
            status = "FAILURE"
        elif performance_ratio > 8 or performance_ratio < -8:
            status = "FAILURE"
        else:
            status = "SUCCESS"

        compare_result_dict.append(
            {
                "model": model,
                "current_accuracy": current_accuracy,
                "last_accuracy": last_accuracy,
                "accuracy_ratio": accuracy_ratio,
                "current_performance": current_performance,
                "last_performance": last_performance,
                "performance_ratio": performance_ratio,
                "status": status,
            }
        )
    return compare_result_dict


def generate(rendered_template):
    with open("generated/report.html", "w") as html_file:
        html_file.write(rendered_template)


def main():
    path = "{}/templates/".format(os.path.dirname(__file__))
    BUILD_BUILDID = os.getenv("BUILD_BUILDID")

    loader = FileSystemLoader(path)
    env = Environment(loader=loader)
    template = env.get_template("model.jinja2")

    data = get_data(args.json_path)
    last_data = get_data(args.last_json_path)
    data = get_accuracy_ratio(data, last_data)
    info = {
        "url": f"https://dev.azure.com/lpot-inc/onnx-neural-compressor/_build/results?buildId={BUILD_BUILDID}",
        "branch": os.getenv("SYSTEM_PULLREQUEST_SOURCEBRANCH"),
        "commit": os.getenv("BUILD_SOURCEVERSION"),
        "build_number": BUILD_BUILDID,
    }

    rendered_template = template.render(data=data, info=info)
    generate(rendered_template)


if __name__ == "__main__":
    main()
