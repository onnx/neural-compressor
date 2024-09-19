# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import tempfile
from typing import Union

import onnx
import onnxruntime as ort
from packaging import version

from onnx_neural_compressor import constants, data_reader, logger, utility
from onnx_neural_compressor.algorithms.post_training_quant import calibrate, quantizer
from onnx_neural_compressor.algorithms.smoother import core
from onnx_neural_compressor.algorithms.weight_only import awq, gptq, rtn
from onnx_neural_compressor.quantization import QuantFormat, config

ort_version = version.Version(ort.__version__)


###################### RTN Algo Entry ##################################
@utility.register_algo(name=constants.RTN)
def rtn_quantize_entry(
    model: Union[pathlib.Path, str], quant_config: config.RTNConfig, *args, **kwargs
) -> onnx.ModelProto:
    """The main entry to apply rtn quantization."""
    config_mapping = quant_config.to_config_mapping(model=model)

    quant_kwargs = dict(
        zip(
            config.RTNConfig.model_params_list,
            [getattr(quant_config, key, None) for key in config.RTNConfig.model_params_list],
        )
    )
    model = rtn.apply_rtn_on_model(model, config_mapping, **quant_kwargs)
    return model


###################### GPTQ Algo Entry ##################################
@utility.register_algo(name=constants.GPTQ)
def gptq_quantize_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.GPTQConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    *args,
    **kwargs,
) -> onnx.ModelProto:
    """The main entry to apply gptq quantization."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/data_reader.py to implement calibration_data_reader"

    config_mapping = quant_config.to_config_mapping(model=model)
    quant_kwargs = dict(
        zip(
            config.GPTQConfig.model_params_list,
            [getattr(quant_config, key, None) for key in config.RTNConfig.model_params_list],
        )
    )

    # regenerate to ensure data exists
    calibration_data_reader.rewind()
    model = gptq.apply_gptq_on_model(model, config_mapping, calibration_data_reader, **quant_kwargs)
    return model


###################### AWQ Algo Entry ##################################
@utility.register_algo(name=constants.AWQ)
def awq_quantize_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.AWQConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    *args,
    **kwargs,
) -> onnx.ModelProto:
    """The main entry to apply awq quantization."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/data_reader.py to implement calibration_data_reader"

    config_mapping = quant_config.to_config_mapping(model=model)
    quant_kwargs = dict(
        zip(
            config.AWQConfig.model_params_list,
            [getattr(quant_config, key, None) for key in config.RTNConfig.model_params_list],
        )
    )

    # regenerate to ensure data exists
    calibration_data_reader.rewind()
    model = awq.apply_awq_on_model(model, config_mapping, calibration_data_reader, **quant_kwargs)
    return model


###################### Static quant Entry ##################################
@utility.register_algo(name=constants.STATIC_QUANT)
def static_quantize_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.StaticQuantConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    model_output: Union[pathlib.Path, str] = None,
    *args,
    **kwargs,
) -> onnx.ModelProto:
    """The main entry to apply dynamic quantization."""
    if len(quant_config.op_types_to_quantize) == 0:
        logger.warning("No candidate op type to do quantization, exit.")
        exit(0)
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/quantization/calibrate.py to implement calibration_data_reader"

    config_mapping = quant_config.to_config_mapping(model=model)

    calibration_data_reader.rewind()
    augment = calibrate.ONNXRTAugment(
        model,
        calibration_data_reader,
        dump_op_types=quant_config.op_types_to_quantize,
        execution_provider=quant_config.execution_provider,
        iterations=list(range(0, quant_config.calibration_sampling_size)),
    )
    min_max = augment.dump_minmax(config_mapping)
    quantize_params = augment.dump_calibration(config_mapping, min_max=min_max)
    _quantizer = quantizer.StaticQuantizer(
        model,
        config_mapping,
        quant_format=quant_config.quant_format.name.lower(),
        quantization_params=quantize_params,
        op_types_to_quantize=quant_config.op_types_to_quantize,
        execution_provider=quant_config.execution_provider,
        optypes_to_exclude_output_quant=quant_config.optypes_to_exclude_output_quant,
        dedicated_qdq_pair=quant_config.dedicated_qdq_pair,
        add_qdq_pair_to_weight=quant_config.add_qdq_pair_to_weight,
    )
    _quantizer.quantize_model()
    if model_output is not None:
        _quantizer.model.save(model_output)
    return _quantizer.model.model


###################### SmoothQuant Entry ##################################
@utility.register_algo(name=constants.SMOOTH_QUANT)
def smooth_quant_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.SmoothQuantConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    model_output: Union[pathlib.Path, str] = None,
    *args,
    **kwargs,
) -> Union[pathlib.Path, str, onnx.ModelProto]:
    """Apply smooth quant."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/data_reader.py to implement calibration_data_reader"

    # smooth operation
    calibration_data_reader.rewind()
    smoother = core.Smoother(
        model,
        calibration_data_reader,
        execution_provider=getattr(quant_config, "execution_provider", "CPUExecutionProvider"),
    )
    smoothed_model = smoother.transform(**quant_config.get_model_params_dict())
    with tempfile.TemporaryDirectory(prefix="ort.quant.") as tmp_dir:
        # ORT quant API requires str input
        onnx.save_model(
            smoothed_model,
            pathlib.Path(tmp_dir).joinpath("smooth.onnx").as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="smooth.onnx_data",
            size_threshold=1024,
            convert_attribute=False,
        )

        # quant operation
        calibration_data_reader.rewind()

        # exclude Mul operations which are inserted during smooth operation
        excluded_nodes = [i.name for i in smoothed_model.graph.node if i.name.endswith("_smooth_mul")]
        quant_config.nodes_to_exclude.extend(excluded_nodes)

        q_model = static_quantize_entry(
            pathlib.Path(tmp_dir).joinpath("smooth.onnx").as_posix(),
            quant_config,
            calibration_data_reader,
            model_output,
        )
    return q_model


###################### Dynamic quant Entry ##################################
@utility.register_algo(name=constants.DYNAMIC_QUANT)
def dynamic_quantize_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.DynamicQuantConfig,
    model_output: Union[pathlib.Path, str] = None,
    *args,
    **kwargs,
) -> onnx.ModelProto:
    """The main entry to apply dynamic quantization."""
    if len(quant_config.op_types_to_quantize) == 0:
        logger.warning("No candidate op type to do quantization, exit.")
        exit(0)

    config_mapping = quant_config.to_config_mapping(model=model)

    _quantizer = quantizer.DynamicQuantizer(
        model,
        config_mapping,
        op_types_to_quantize=quant_config.op_types_to_quantize,
    )
    _quantizer.quantize_model()
    if model_output is not None:
        _quantizer.model.save(model_output)
    return _quantizer.model.model
