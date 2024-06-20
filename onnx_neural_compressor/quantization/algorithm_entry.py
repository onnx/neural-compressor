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
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.post_training_quant import calibrate
from onnx_neural_compressor.algorithms.post_training_quant import quantizer
from onnxruntime import quantization

from onnx_neural_compressor import config, constants, data_reader, logger, utility
from onnx_neural_compressor.algorithms.smoother import core
from onnx_neural_compressor.algorithms.weight_only import awq, gptq, rtn


###################### RTN Algo Entry ##################################
@utility.register_algo(name=constants.RTN)
def rtn_quantize_entry(
    model: Union[pathlib.Path, str], quant_config: config.RTNConfig, *args, **kwargs
) -> onnx.ModelProto:
    """The main entry to apply rtn quantization."""
    if len(quant_config.config_mapping) == 0:
        # map config to each op
        model_info = config.RTNConfig.get_model_info(model=model)
        config_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.debug(config_mapping)
    else:
        config_mapping = quant_config.config_mapping
    model = rtn.apply_rtn_on_model(model, config_mapping)
    quant_utils.dump_woq_stats(model, config_mapping, quant_config.white_list)
    return model


###################### GPTQ Algo Entry ##################################
@utility.register_algo(name=constants.GPTQ)
def gptq_quantize_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.GPTQConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    *args,
    **kwargs
) -> onnx.ModelProto:
    """The main entry to apply gptq quantization."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/data_reader.py to implement calibration_data_reader"

    if len(quant_config.config_mapping) == 0:
        # map config to each op
        model_info = config.GPTQConfig.get_model_info(model=model)
        config_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.debug(config_mapping)
    else:
        config_mapping = quant_config.config_mapping

    # regenerate to ensure data exists
    calibration_data_reader.rewind()
    model = gptq.apply_gptq_on_model(model, config_mapping, calibration_data_reader)
    quant_utils.dump_woq_stats(model, config_mapping, quant_config.white_list)
    return model


###################### AWQ Algo Entry ##################################
@utility.register_algo(name=constants.AWQ)
def awq_quantize_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.AWQConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    *args,
    **kwargs
) -> onnx.ModelProto:
    """The main entry to apply awq quantization."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"
    assert isinstance(
        calibration_data_reader, data_reader.CalibrationDataReader
    ), "Please follow onnx_neural_compressor/data_reader.py to implement calibration_data_reader"

    if len(quant_config.config_mapping) == 0:
        # map config to each op
        model_info = config.AWQConfig.get_model_info(model=model)
        config_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.debug(config_mapping)
    else:
        config_mapping = quant_config.config_mapping

    # regenerate to ensure data exists
    calibration_data_reader.rewind()
    model = awq.apply_awq_on_model(model, config_mapping, calibration_data_reader)
    quant_utils.dump_woq_stats(model, config_mapping, quant_config.white_list)
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

    if len(quant_config.config_mapping) == 0:
        # map config to each op
        model_info = config.StaticQuantConfig.get_model_info(model=model)
        config_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.debug(config_mapping)
    else:
        config_mapping = quant_config.config_mapping

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
        optypes_to_exclude_output_quant=quant_config.extra_options.get("optypes_to_exclude_output_quant", []),
    )
    _quantizer.quantize_model()
    if model_output is not None:
        _quantizer.model.save(model_output)
    quant_utils.dump_model_op_stats(_quantizer.model.model, config_mapping, quant_config.op_types_to_quantize)
    return _quantizer.model.model


###################### SmoothQuant Entry ##################################
@utility.register_algo(name=constants.SMOOTH_QUANT)
def smooth_quant_entry(
    model: Union[pathlib.Path, str],
    quant_config: config.SmoothQuantConfig,
    calibration_data_reader: data_reader.CalibrationDataReader,
    model_output: Union[pathlib.Path, str] = None,
    *args,
    **kwargs
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
        execution_provider=getattr(quant_config, "execution_provider", "CPUExecutionProvider")
    )
    smoothed_model = smoother.transform(**quant_config.to_dict())
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

    if len(quant_config.config_mapping) == 0:
        # map config to each op
        model_info = config.DynamicQuantConfig.get_model_info(model=model)
        config_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.debug(config_mapping)
    else:
        config_mapping = quant_config.config_mapping

    _quantizer = quantizer.DynamicQuantizer(
        model,
        config_mapping,
        op_types_to_quantize=quant_config.op_types_to_quantize,
        )
    _quantizer.quantize_model()
    if model_output is not None:
        _quantizer.model.save(model_output)
    quant_utils.dump_model_op_stats(_quantizer.model.model, config_mapping, quant_config.op_types_to_quantize)
    return _quantizer.model.model
