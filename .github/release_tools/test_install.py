# Copyright (c) ONNX Neural Compressor Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Install each artifact in its own clean environment and test outside the source tree."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("version")
    parser.add_argument("--constraints", type=Path, help="Test with pinned minimum runtime dependencies")
    args = parser.parse_args()
    constraints = ["--constraint", str(args.constraints.resolve())] if args.constraints else []
    smoke = Path(__file__).with_name("smoke.py").resolve()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for pattern in ("*.whl", "*.tar.gz"):
        artifacts = list(args.directory.resolve().glob(pattern))
        if len(artifacts) != 1:
            raise ValueError(f"Expected one {pattern} artifact")
        with tempfile.TemporaryDirectory(prefix="onc-release-") as directory:
            root = Path(directory)
            subprocess.run([sys.executable, "-m", "venv", str(root / "venv")], check=True, env=environment)
            python = root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            commands = [
                [str(python), "-m", "pip", "install", "--no-cache-dir", *constraints, str(artifacts[0])],
                [str(python), "-m", "pip", "check"],
                [str(python), "-m", "pip", "freeze"],
                [str(python), str(smoke), args.version],
            ]
            for command in commands:
                subprocess.run(command, cwd=root, env=environment, check=True, timeout=600)


if __name__ == "__main__":
    main()
