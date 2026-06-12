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
"""Calibration for smooth quant."""

import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
from typing import List

import numpy as np
import onnx
import onnxruntime
import tqdm

from onnx_neural_compressor import data_reader, logger, onnx_model, utility
from onnx_neural_compressor.algorithms import utility as quant_utils


def _acts_dir(checkpoint_dir):
    return pathlib.Path(checkpoint_dir) / "smooth-acts"


def clear_acts_checkpoint(checkpoint_dir):
    """Remove the partial per-sample activation dumps. Called once the FULL smooth-calib
    checkpoint exists (the per-sample files are superseded, and they are the bulky part:
    every smoothed tensor's activation for every collected sample)."""
    shutil.rmtree(_acts_dir(checkpoint_dir), ignore_errors=True)


def load_acts_checkpoint(checkpoint_dir, keys):
    """Restore the per-sample activation dumps of an interrupted calibration pass.

    Returns a list of {tensor_name: array} dicts for the contiguous samples 0..k-1
    found in <checkpoint_dir>/smooth-acts/. The dumped tensor set must match `keys`
    exactly (same op_types config); on mismatch the stale dump is deleted and []
    returned, so a changed configuration can never half-resume into wrong numerics."""
    d = _acts_dir(checkpoint_dir)
    names_file = d / "names.json"
    if not names_file.exists():
        return []
    with open(names_file) as f:
        names = json.load(f)
    if names != list(keys):
        logger.warning(
            "smooth-acts checkpoint in {} was dumped for a different tensor set; discarding it".format(d)
        )
        clear_acts_checkpoint(checkpoint_dir)
        return []
    samples = []
    while True:
        f = d / "sample-{:05d}.npz".format(len(samples))
        if not f.exists():
            break
        with np.load(f) as z:
            samples.append({n: z["arr_{}".format(j)] for j, n in enumerate(names)})
    return samples


def save_acts_sample(checkpoint_dir, keys, index, sample):
    """Dump one collected sample's activations (atomically) as smooth-acts/sample-NNNNN.npz,
    arrays positional in the order of `keys` (recorded once in names.json: npz keys cannot
    safely hold every ONNX tensor-name character)."""
    d = _acts_dir(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    names_file = d / "names.json"
    if not names_file.exists():
        tmp = str(names_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(list(keys), f)
        os.replace(tmp, names_file)
    path = d / "sample-{:05d}.npz".format(index)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, *[sample[k] for k in keys])
    os.replace(tmp, path)


class Calibrator:
    """Dump information for smooth quant."""

    def __init__(
        self,
        model: onnx_model.ONNXModel,
        dataloader: data_reader.CalibrationDataReader,
        iterations: List[int] = [],
        providers: List[str] = ["CPUExecutionProvider"],
        checkpoint_dir=None,
        checkpoint_interval_sec: float = 1200,
        **kwargs,
    ):
        """Initialize a Calibrator to dump information.

        Args:
            model (onnx_model.ONNXModel): onnx_model.ONNXModel object.
            dataloader (data_reader.CalibrationDataReader): user implemented object to read in and preprocess calibration dataset.
            iterations (List[int], optional): tensor of which iteration will be collected. Defaults to [].
            providers (List[str], optional): execution provider for onnxruntime. Defaults to ["CPUExecutionProvider"].
            checkpoint_dir (str, optional): when set, the collected per-sample activations
                are periodically dumped under <checkpoint_dir>/smooth-acts/ and reloaded on
                the next run, so an interrupted calibration pass resumes mid-pass instead of
                redoing every forward.
            checkpoint_interval_sec (float, optional): minimum seconds between those dumps
                (they are activation-sized, so a fast pass should not pay the IO; 0 dumps
                after every sample). Defaults to 1200 (20 minutes).
        """
        self.model_wrapper = model
        self.dataloader = dataloader
        self.augmented_model = None
        self.iterations = iterations
        self.providers = providers
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval_sec = checkpoint_interval_sec

    def _check_is_group_conv(self, node):
        """Check the op is group wised or not(depthwise conv is excluded,return false).

        Args:
            node: The op node

        Returns:
            Bool: group wised True, otherwise False, depthwise False
        """
        name_to_indices = {}
        for index, i in enumerate(self.model_wrapper.initializer()):
            name_to_indices[i.name] = index

        if node.op_type == "Conv":
            group = 1
            for attr in node.attribute:
                if hasattr(attr, "name"):
                    if attr.name == "group":
                        group = attr.i
                        break
            # currently only normal conv and depthwise conv are supported
            if group > 1:  # group conv, need to check depthwise or not
                weight_name = node.input[1]
                weight_shape = onnx.numpy_helper.to_array(
                    self.model_wrapper.initializer()[name_to_indices[weight_name]]
                ).shape
                input_channel = weight_shape[1]
                if input_channel != 1:  # TODO: need to double check
                    return True
        return False

    def _get_input_tensor_of_ops(self, op_types: List[str] = ["MatMul", "Gemm", "Conv", "FusedConv"]):
        """Traverse the graph and get all the data tensors flowing into layers of {op_types}.

        Group conv is excluded.
        # TODO: the tensors could be set/filtered in configuration.

        Args:
            op_types (List[str], optional): The op types whose input tensor will be dumped.
                Defaults to ["MatMul", "Gemm", "Conv", "FusedConv"].

        Returns:
            dict: A dict of dumped tensor to node info
        """
        tensors_to_node = {}
        initializers = {i.name: i for i in self.model_wrapper.initializer()}

        for node in self.model_wrapper.nodes():
            if len(op_types) == 0 or node.op_type in op_types:
                if node.op_type in ["Conv", "FusedConv"] and self._check_is_group_conv(node):
                    continue
                # also need to check whether the layer has weight
                if len(node.input) >= 2 and node.input[1] in initializers.keys():
                    tensors_to_node.setdefault(node.input[0], []).append([node.name, node.input, node.output])
        return tensors_to_node

    def _get_max_per_channel(self, datas, percentile):
        """Get the max values per input channel.

        Args:
            datas: The tensors
            percentile: percentile of calibration to remove outliers

        Returns:
            The max values per input channel
        """
        permute_datas = []
        for data in datas:
            if len(data.shape) == 3:  # TODO: mammul batchsize*seq*inchannel, conv:batchsize*inchannle*f*f
                tensor = np.abs(np.reshape(data, (-1, data.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 4:
                tensor = np.swapaxes(data, 1, -1)
                tensor = np.abs(np.reshape(tensor, (-1, tensor.shape[-1])))
                permute_datas.append(tensor)
            elif len(data.shape) == 2:
                permute_datas.append(np.abs(data))
            else:
                assert False, "not supported"
        permute_datas = np.stack(permute_datas, axis=0)
        permute_datas = permute_datas.reshape(-1, permute_datas.shape[-1])
        max_per_channels = np.percentile(permute_datas, percentile, axis=0)
        max_per_channels = max_per_channels.astype(np.single)
        return max_per_channels

    def get_intermediate_outputs(self, checkpoint_keys=None):
        so = onnxruntime.SessionOptions()
        if sys.version_info < (3, 11) and importlib.util.find_spec("onnxruntime_extensions"):  # pragma: no cover
            from onnxruntime_extensions import get_library_path

            so.register_custom_ops_library(get_library_path())

        providers = self.providers if "TensorrtExecutionProvider" not in self.providers else ["CUDAExecutionProvider"]
        providers = quant_utils.conservative_session_resources(so, providers)
        if self.model_wrapper.is_large_model:  # pragma: no cover
            with tempfile.TemporaryDirectory(prefix="ort.calib.") as tmp_dir:
                onnx.save_model(
                    self.model_wrapper.model,
                    pathlib.Path(tmp_dir).joinpath("augment.onnx").as_posix(),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                )
                session = onnxruntime.InferenceSession(
                    pathlib.Path(tmp_dir).joinpath("augment.onnx").as_posix(), so, providers=providers
                )

                onnx.external_data_helper.load_external_data_for_model(
                    self.model_wrapper.model, pathlib.Path(tmp_dir).as_posix()
                )
        else:
            session = onnxruntime.InferenceSession(
                self.model_wrapper.model.SerializeToString(), so, providers=providers
            )
        node_output_names = [output.name for output in session.get_outputs()]
        output_dicts = {}
        input_name_to_nodes = self.model_wrapper.input_name_to_nodes()
        output_name_to_node = self.model_wrapper.output_name_to_node()
        name_to_node = {}
        for data_name in node_output_names:
            node = None
            if data_name in output_name_to_node:
                node = output_name_to_node[data_name]
            elif data_name in input_name_to_nodes:
                node = input_name_to_nodes[data_name][0]
            assert node, "{} is neither an input nor an output of nodes in augmented model.".format(data_name)
            name_to_node[data_name] = node.name

        def _collect_data(ort_inputs):
            for output_idx, output in enumerate(session.run(None, ort_inputs)):
                output_dicts.setdefault(node_output_names[output_idx], []).append(output)

        # Mid-pass resume: samples already dumped by an interrupted run are restored from
        # the checkpoint and their forwards skipped entirely. Only checkpoint_keys (the
        # smoothed tensors, the only ones calib_smooth reads) are dumped/restored, not the
        # model's real outputs. Dumps are time-gated (the state is activation-sized): on a
        # fast pass nothing is ever written; on a slow pass at most checkpoint_interval_sec
        # of forwards can be lost to a crash.
        checkpoint_on = bool(self.checkpoint_dir and checkpoint_keys)
        restored = 0
        if checkpoint_on:
            for sample in load_acts_checkpoint(self.checkpoint_dir, checkpoint_keys):
                for name, arr in sample.items():
                    output_dicts.setdefault(name, []).append(arr)
                restored += 1
            if restored:
                logger.info(
                    "smooth-acts checkpoint: restored {} collected sample(s); "
                    "skipping their forwards".format(restored)
                )
        state = {"collected": 0, "pending": [], "last_flush": time.time(), "flushed": False}

        def _flush_pending():
            for index, sample in state["pending"]:
                save_acts_sample(self.checkpoint_dir, checkpoint_keys, index, sample)
            state["pending"].clear()
            state["last_flush"] = time.time()
            state["flushed"] = True

        def _consume(ort_inputs):
            if state["collected"] < restored:
                state["collected"] += 1  # already restored from the checkpoint
                return
            _collect_data(ort_inputs)
            if checkpoint_on:
                state["pending"].append(
                    (state["collected"], {k: output_dicts[k][-1] for k in checkpoint_keys})
                )
                if time.time() - state["last_flush"] >= self.checkpoint_interval_sec:
                    _flush_pending()
            state["collected"] += 1

        # This per-sample forward pass over the calibration set is the slow, otherwise
        # silent phase logged as "Start smooth model calibration"; show its progress.
        total = (max(self.iterations) + 1) if self.iterations else None
        pbar = tqdm.tqdm(total=total, desc="SmoothQuant: collecting calibration activations",
                         unit="sample", leave=False)
        idx = 0
        while True:
            inputs = self.dataloader.get_next()
            if not inputs:
                break
            if self.iterations != []:
                if idx > max(self.iterations):
                    break
                if idx in self.iterations:
                    _consume(inputs)
            else:
                _consume(inputs)
            idx += 1
            pbar.update(1)
        pbar.close()
        # Flush the tail only when a periodic flush already happened: a pass slow enough
        # to have flushed deserves a complete checkpoint (the np.percentile reduction and
        # the smooth-calib save still lie ahead and can OOM), while a fast pass should
        # not suddenly write gigabytes that the full smooth-calib checkpoint supersedes
        # moments later.
        if checkpoint_on and state["flushed"] and state["pending"]:
            _flush_pending()
        return output_dicts

    def calib_smooth(self, op_types, percentile: float = 99.999):
        """Smooth model calibration.

        Mainly get the max info per channel of input tensors.

        Args:
            op_types (_type_): The op types whose input tensor will be dumped.
            percentile (float, optional): Percentile of calibration to remove outliers.
                Defaults to 99.999.

        Returns:
            max_vals_per_channel: max values per channel of input tensors
            shape_infos: The shape information of input tensors
        """
        logger.info("Start smooth model calibration.")
        # add the input tensors of {op_types} to outputs of the model
        tensors_to_node = self._get_input_tensor_of_ops(op_types)
        self.model_wrapper.add_tensors_to_outputs(tensors_to_node.keys())
        output_dicts = self.get_intermediate_outputs(checkpoint_keys=list(tensors_to_node.keys()))

        # remove the input tensors of {op_types} to outputs of the model
        self.model_wrapper.remove_tensors_from_outputs(tensors_to_node.keys())
        max_vals_per_channel = {}
        shape_infos = {}

        for key, val in tensors_to_node.items():
            max_val_per_channel = self._get_max_per_channel(output_dicts[key], percentile=percentile)
            max_vals_per_channel[key] = max_val_per_channel
            shape_infos[key] = output_dicts[key][0].shape
            for item in val:
                shape_infos[item[1][1]] = self.model_wrapper.get_initializer(item[1][1]).dims
        return max_vals_per_channel, shape_infos, tensors_to_node
