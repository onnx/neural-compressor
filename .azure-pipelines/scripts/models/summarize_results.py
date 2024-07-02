import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--logs_dir", required=True, type=str)
parser.add_argument("--output_dir", required=True, type=str)
args = parser.parse_args()


def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def write_json_file(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def merge_json_files(root_dir, output_file):
    merged_data = {}

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                try:
                    json_data = read_json_file(file_path)
                    merged_data.update(json_data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {file_path}")

    print(merged_data)
    write_json_file(merged_data, f"{output_file}/summary.json")


def main():
    merge_json_files(args.logs_dir, args.output_dir)
    print(f"All JSON files have been merged into {args.output_dir}")


if __name__ == "__main__":
    main()
