# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import pathlib
import tempfile

import onnx
import onnxruntime as ort

from onnx_neural_compressor import data_reader, logger, onnx_model

from typing import Callable, List, Union  # isort: skip


def layer_wise_quant(
    model: Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str],
    quant_func: Callable,
    weight_config: dict,
    data_reader: data_reader.CalibrationDataReader = None,
    *args,
    **kwargs
) -> onnx_model.ONNXModel:
    """Quantize model layer by layer to save memory.

    Args:
        model (Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str]): onnx model.
        quant_func (Callable): quantization algo function.
        weight_config (dict): quantization config.
        data_reader (data_reader.CalibrationDataReader, optional): data_reader for calibration. Defaults to None.

    Returns:
        _type_: _description_
    """
    logger.warning(
        "Layer-wise quantization requires data_type info for some tensors. "
        "We will try to infer the data_type automatically if it doesn't exist."
        "You can use model with symbolic shape inference before layer-wise quantization as well like follows:\n"
        "import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer\n"
        "model = onnx.load(your_model_path)\n"
        "out = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model, auto_merge=True)\n"
        "onnx.save_model(out, infer_shape_model_path, save_as_external_data=True)\n"
    )

    if not isinstance(model, onnx_model.ONNXModel):
        model = onnx_model.ONNXModel(model, ignore_warning=True, load_external_data=False)

    origin_model = copy.deepcopy(model)
    tmp_file = tempfile.TemporaryDirectory()
    providers = kwargs.get("providers", ["CPUExecutionProvider"])

    # get and check split nodes
    split_nodes = origin_model.find_split_nodes()
    if len(split_nodes) == 0:
        logger.error("Can't find split nodes for layer-wise quantization.")
        raise ValueError("Fail to run layer-wise quantization.")
    logger.info(
        "Will split model into {} parts to do layer-wise quantization".format(
            len([node.name for node in split_nodes]) + 1
        )
    )
    logger.debug(
        "Will split model with these nodes for layer-wise quantization: {}".format([node.name for node in split_nodes])
    )

    split_idx = 1
    model_to_split = [origin_model]
    quantized_model_merged = None

    require_data_reader = data_reader is not None
    if require_data_reader:
        lwq_data_reader = [data_reader]

    while len(model_to_split) != 0:
        # prepare model, node and data_reader for current split
        split_model = model_to_split.pop(0)
        split_node = split_nodes.pop(0)
        if require_data_reader:
            complete_data_reader = lwq_data_reader.pop(0)

        # if no remaining split nodes, it means this is the last split, and the two split models will be saved.
        save_both_split_models = True if len(split_nodes) == 0 else False

        # split model with given split node
        split_model_part_1, split_model_part_2 = split_model.split_model_with_node(
            split_node.name, model.model_path, save_both_split_models, save_path=tmp_file.name
        )

        if not save_both_split_models:
            # append split_model_part_2 to do next split
            model_to_split.append(split_model_part_2)

        logger.info("Quantize split model {}".format(split_idx))

        if require_data_reader:
            # process data_reader for current split and next split
            current_data_reader = _filter_data_reader_for_current_split_model(
                split_model_part_1.model, complete_data_reader
            )

            # complete_data_reader contains split_model_part_1 output data
            complete_data_reader = _prepare_data_reader_for_next_split_model(
                split_model_part_1.model_path,
                [i.name for i in split_model_part_2.model.graph.input],
                complete_data_reader,
                providers,
            )

            lwq_data_reader.append(complete_data_reader)

            # perform quantization
            split_model_part_1_quantized = quant_func(
                split_model_part_1,
                weight_config=weight_config,
                data_reader=current_data_reader,
                return_modelproto=False,
                **kwargs
            )
        else:
            # perform quantization
            split_model_part_1_quantized = quant_func(
                split_model_part_1, weight_config=weight_config, return_modelproto=False, **kwargs
            )

        # check split model is valid
        try:
            ort.InferenceSession(split_model_part_1_quantized.model_path, providers=providers)
        except Exception as e:
            logger.error(
                "Layer-wise quantized model {} can't be inferred correctly. "
                "Please check the raise exception".format(split_idx)
            )
            raise e

        # merge split quantized model
        if quantized_model_merged is None:
            quantized_model_merged = split_model_part_1_quantized
            quantized_model_merged.write_external_data_to_new_location(overwrite=True)
        else:
            quantized_model_merged.merge_split_models(split_model_part_1_quantized)

        split_idx += 1
        # if this is the last split, quantize the last split model
        if save_both_split_models:
            logger.info("Quantize split model {}".format(split_idx))

            # quantize split model
            if require_data_reader:
                # process data_reader for current split
                current_data_reader = lwq_data_reader.pop(0)
                current_data_reader = _filter_data_reader_for_current_split_model(
                    split_model_part_2.model, complete_data_reader
                )

                # perform quantization
                split_model_part_2_quantized = quant_func(
                    split_model_part_2,
                    weight_config=weight_config,
                    data_reader=current_data_reader,
                    return_modelproto=False,
                    **kwargs
                )
            else:
                # perform quantization
                split_model_part_2_quantized = quant_func(
                    split_model_part_2, weight_config=weight_config, return_modelproto=False, **kwargs
                )

            # check split model is valid
            try:
                ort.InferenceSession(split_model_part_2_quantized.model_path, providers=providers)
            except Exception as e:
                logger.error(
                    "Layer-wise quantized model {} can't be inferred correctly. "
                    "Please check the raise exception".format(split_idx)
                )
                raise e

            # merge split quantized model
            if quantized_model_merged is None:
                quantized_model_merged = split_model_part_2_quantized
                quantized_model_merged.write_external_data_to_new_location(overwrite=True)
            else:
                quantized_model_merged.merge_split_models(split_model_part_2_quantized)

    # reload external data to prevent external data file path errors
    onnx.external_data_helper.load_external_data_for_model(
        quantized_model_merged.model, os.path.dirname(quantized_model_merged.model_path)
    )

    tmp_file.cleanup()
    return quantized_model_merged


class DataReader(data_reader.CalibrationDataReader):
    """Data reader for layer-wise quantization."""

    def __init__(self, data_list):
        self.data_list = data_list
        self.iter_next = iter(self.data_list)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.data_list)


def _filter_data_reader_for_current_split_model(
    model: onnx.ModelProto,
    current_data_reader: data_reader.CalibrationDataReader,
):
    """Filter data reader to remove data that is not in model input.

    Args:
        model (onnx.ModelProto): onnx model.
        current_data_reader (data_reader.CalibrationDataReader): data reader of current split model.

    Returns:
        data_reader.CalibrationDataReader: filtered data reader.
    """
    filter_inputs = []
    input_names = [input.name for input in model.graph.input]
    current_data_reader.rewind()

    while True:
        inputs = current_data_reader.get_next()
        if not inputs:
            break
        filter_input = {
            input_name: input_tensor for input_name, input_tensor in inputs.items() if input_name in input_names
        }
        filter_inputs.append(filter_input)

    return DataReader(filter_inputs)


def _prepare_data_reader_for_next_split_model(
    model_path: str,
    next_model_input_names: list,
    data_reader: data_reader.CalibrationDataReader,
    providers: List[str] = ["CPUExecutionProvider"],
):
    """Prepare data reader for next split model.

    Get data output of current split model and save for next split model.

    Args:
        model (str): path to onnx model.
        data_reader (data_reader.CalibrationDataReader): data reader
        providers (List[str], optional): providers to use. Defaults to ["CPUExecutionProvider"].

    Returns:
        data_reader.CalibrationDataReader: data reader for next split model.
    """
    data_reader.rewind()
    data_reader_for_next_split_model = []
    session = ort.InferenceSession(model_path, providers=providers)
    output_names = [output.name for output in session.get_outputs()]
    input_names = [input.name for input in session.get_inputs()]
    while True:
        inputs = data_reader.get_next()
        if not inputs:
            break
        out = session.run(None, {name: inputs[name] for name in input_names})
        filter_input = {name: value for name, value in zip(output_names, out)}
        for name, value in inputs.items():
            if name in next_model_input_names and name not in filter_input:
                filter_input[name] = value
        data_reader_for_next_split_model.append(filter_input)
    return DataReader(data_reader_for_next_split_model)
