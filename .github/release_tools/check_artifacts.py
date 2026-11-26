# Copyright (c) ONNX Neural Compressor Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check that the wheel and source distribution carry the requested version."""

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def check_metadata(data, version):
    metadata = BytesParser().parsebytes(data)
    if metadata["Name"].replace("_", "-") != "onnx-neural-compressor" or metadata["Version"] != version:
        raise ValueError(f"Unexpected distribution: {metadata['Name']} {metadata['Version']}, expected {version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()
    wheels = list(args.directory.glob("*.whl"))
    sources = list(args.directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected one wheel METADATA file")
        check_metadata(wheel.read(metadata[0]), args.version)
    with tarfile.open(sources[0], "r:gz") as source:
        root = sources[0].name.removesuffix(".tar.gz")
        check_metadata(source.extractfile(f"{root}/PKG-INFO").read(), args.version)
        source_version = source.extractfile(f"{root}/VERSION_NUMBER").read().decode("utf-8").strip()
        if source_version != args.version:
            raise ValueError("Source VERSION_NUMBER differs from the distribution metadata")
    print(f"Wheel and source distribution both contain {args.version}")


if __name__ == "__main__":
    main()
