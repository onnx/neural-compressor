"""Shared pytest configuration."""

import sys

# Released Optimum/Optimum-ONNX packages do not yet support Python 3.14:
# functools.partial exporter configuration attributes bind as methods there.
# The portable suite and release quantization smoke tests still cover 3.14.
if sys.version_info >= (3, 14):
    collect_ignore = [
        "quantization/layer_wise/test_layer_wise.py",
        "quantization/post_training_quant/test_post_training_quant.py",
        "quantization/test_autotune.py",
        "quantization/test_config.py",
        "quantization/test_smooth_quant.py",
        "quantization/weight_only/test_awq.py",
        "quantization/weight_only/test_gptq.py",
        "quantization/weight_only/test_rtn.py",
        "utils/test_onnx_model.py",
    ]
