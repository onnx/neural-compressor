import re
import subprocess
from pathlib import Path

from setuptools import setup

version_file = Path(__file__).parent / "onnx_neural_compressor" / "version.py"
version_match = re.search(r'^__version__ = "([^"]+)"$', version_file.read_text(encoding="utf-8"), re.MULTILINE)
if version_match is None:
    raise RuntimeError(f"Could not find package version in {version_file}")
base_version = version_match.group(1)


def get_build_version():
    try:
        subprocess.run(["git", "describe", "--exact-match", "--tags"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        return base_version

    try:
        description = subprocess.run(
            ["git", "describe", "--tags"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return base_version

    match = re.fullmatch(r".+-(\d+)-(g[0-9a-fA-F]+)", description)
    if match is None:
        raise RuntimeError(f"Unexpected output from `git describe --tags`: {description!r}")
    return f"{base_version}.dev{match.group(1)}+{match.group(2)}"


setup(version=get_build_version())
