# Copyright (c) ONNX Neural Compressor Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare an explicit preview or release version, without consulting Git tags."""

import argparse
import datetime
import os
import re
from pathlib import Path

RELEASE_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:rc([1-9]\d*))?")


def prepare_version(version, mode, publish, event, repository, ref, today):
    match = RELEASE_VERSION.fullmatch(version)
    if match is None:
        raise ValueError("VERSION_NUMBER must contain X.Y.Z or X.Y.ZrcN (N >= 1)")
    if mode not in ("preview", "release") or publish not in ("none", "testpypi", "pypi"):
        raise ValueError("Invalid build mode or publication target")
    if publish != "none":
        expected_ref = "refs/heads/rel-" + ".".join(match.group(1, 2, 3))
        if mode != "release" or event != "workflow_dispatch":
            raise ValueError("Publishing requires a manually dispatched release build")
        if repository != "onnx/neural-compressor" or ref != expected_ref:
            raise ValueError(f"Publishing requires onnx/neural-compressor on {expected_ref}")
    if mode == "preview":
        return f"{version}.dev{today:%Y%m%d}"
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preview", "release"], default="preview")
    parser.add_argument("--publish", choices=["none", "testpypi", "pypi"], default="none")
    parser.add_argument("--version-file", type=Path, default=Path("VERSION_NUMBER"))
    args = parser.parse_args()
    version = prepare_version(
        args.version_file.read_text(encoding="utf-8").strip(),
        args.mode,
        args.publish,
        os.environ.get("GITHUB_EVENT_NAME", ""),
        os.environ.get("GITHUB_REPOSITORY", ""),
        os.environ.get("GITHUB_REF", ""),
        datetime.datetime.now(datetime.timezone.utc).date(),
    )
    args.version_file.write_text(version + "\n", encoding="utf-8")
    print(version)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"version={version}\n")


if __name__ == "__main__":
    main()
