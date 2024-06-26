import io
import re
import subprocess

import setuptools


def is_commit_on_tag():
    try:
        result = subprocess.run(
            ["git", "describe", "--exact-match", "--tags"], capture_output=True, text=True, check=True
        )
        tag_name = result.stdout.strip()
        return tag_name
    except subprocess.CalledProcessError:
        return False


def get_build_version():
    if is_commit_on_tag():
        return __version__
    try:
        result = subprocess.run(["git", "describe", "--tags"], capture_output=True, text=True, check=True)
        _, distance, commit = result.stdout.strip().split("-")
        return f"{__version__}.dev{distance}+{commit}"
    except subprocess.CalledProcessError:
        return __version__


try:
    filepath = "./onnx_neural_compressor/version.py"
    with io.open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)

if __name__ == "__main__":

    setuptools.setup(
        name="onnx_neural_compressor",
        author="Intel AIPT Team",
        version=get_build_version(),
        author_email="tai.huang@intel.com, mengni.wang@intel.com, yuwen.zhou@intel.com, suyue.chen@intel.com",
        description="Repository of Neural Compressor ORT",
        long_description=io.open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="quantization",
        license="Apache 2.0",
        url="",
        packages=setuptools.find_packages(),
        include_package_data=True,
        install_requires=[
            "onnx",
            "onnxruntime",
            "onnxruntime-extensions",
            "psutil",
            "numpy<2.0.0",
            "py-cpuinfo",
            "pydantic",
            "transformers",
        ],
        python_requires=">=3.8.0",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: Apache Software License",
        ],
    )
