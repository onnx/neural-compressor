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
import math
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


STREAM_STATE_FILE = "smooth-stream.npz"


def clear_stream_checkpoint(checkpoint_dir):
    """Remove the mid-pass calibration state. Called once the FULL smooth-calib
    checkpoint exists (which supersedes it). Also removes the smooth-acts/ directory
    a pre-streaming version of this code may have left behind (it dumped raw
    per-sample activations there; the streaming reducer made that obsolete)."""
    try:
        os.remove(pathlib.Path(checkpoint_dir) / STREAM_STATE_FILE)
    except FileNotFoundError:
        pass
    shutil.rmtree(pathlib.Path(checkpoint_dir) / "smooth-acts", ignore_errors=True)


class StreamingChannelPercentile:
    """Exact streaming replacement for the hold-everything per-channel percentile.

    The smoother needs, per smoothed tensor, the per-channel {percentile} of
    |activations| over every calibration sample. Upstream stacked every sample in
    RAM and called np.percentile once at the end, so memory grew as
    samples x frames x channels (~6 GB per 395 s window on the 0.6B encoder,
    hundreds of GB per export). But for the high percentiles SmoothQuant uses
    (99.999 by default) the answer only depends on the few largest values per
    channel: np.percentile's linear interpolation reads the two order statistics
    around rank (n-1)*q/100, which sit within the top ceil((n-1)*(1-q/100))+1
    values. So this keeps a per-channel running top-K plus the exact row count n,
    folds each sample in as it is collected (which can then be freed), and
    reproduces np.percentile's result bit-for-bit at the end.

    K is sized once, from the planned sample count and the first sample's row
    count with a 4x row headroom; `result` re-derives the rank it needs from the
    ACTUAL n and raises if K turned out too small (wildly varying sample sizes),
    so a result can never be silently wrong. Percentiles low enough to make K
    huge (the streaming advantage vanishes) are rejected up front.
    """

    MAX_K = 8192

    def __init__(self, percentile, planned_samples=None):
        self.percentile = float(percentile)
        # planned_samples=None means "unbounded dataloader": size K for 1024
        # samples and let `result` catch the (pathological) overflow.
        self.planned_samples = planned_samples
        self.k = None
        self.state = {}  # name -> {"topk": (<=K, C) array, "n": int, "shape": tuple}

    @staticmethod
    def _to_rows(data):
        """|data| reshaped to (rows, channels); same layouts as _get_max_per_channel."""
        if len(data.shape) == 3:
            return np.abs(np.reshape(data, (-1, data.shape[-1])))
        if len(data.shape) == 4:
            tensor = np.swapaxes(data, 1, -1)
            return np.abs(np.reshape(tensor, (-1, tensor.shape[-1])))
        if len(data.shape) == 2:
            return np.abs(data)
        assert False, "not supported"

    def _needed_k(self, n):
        return int(math.ceil((n - 1) * (1.0 - self.percentile / 100.0))) + 1

    def _size_k(self, first_sample_rows):
        planned = self.planned_samples if self.planned_samples else 1024
        est_n = max(first_sample_rows, 1) * planned * 4
        k = self._needed_k(est_n) + 8
        if k > self.MAX_K:
            raise ValueError(
                "percentile {} over an estimated {} rows needs a top-{} per channel, "
                "beyond the streaming reducer's cap of {}; use a higher percentile or "
                "fewer/shorter calibration samples".format(self.percentile, est_n, k, self.MAX_K)
            )
        return k

    def add(self, name, data):
        data = np.asarray(data)
        rows = self._to_rows(data)
        if self.k is None:
            self.k = self._size_k(rows.shape[0])
        st = self.state.get(name)
        if st is None:
            st = self.state[name] = {
                "topk": np.empty((0, rows.shape[-1]), dtype=rows.dtype),
                "n": 0,
                "shape": tuple(data.shape),
            }
        st["n"] += rows.shape[0]
        if rows.shape[0] > self.k:
            rows = np.partition(rows, rows.shape[0] - self.k, axis=0)[rows.shape[0] - self.k :]
        merged = np.concatenate([st["topk"], rows], axis=0)
        if merged.shape[0] > self.k:
            merged = np.partition(merged, merged.shape[0] - self.k, axis=0)[merged.shape[0] - self.k :]
        st["topk"] = merged

    def shape(self, name):
        """Shape of the first collected sample for this tensor."""
        return self.state[name]["shape"]

    def result(self, name):
        """The per-channel percentile, exactly as np.percentile(stacked, q, axis=0)."""
        st = self.state[name]
        n = st["n"]
        if n == 0:
            raise RuntimeError("no samples were collected for tensor {}".format(name))
        sbuf = np.sort(st["topk"], axis=0)  # ascending per channel, the global top-m
        m = sbuf.shape[0]
        virtual = self.percentile / 100.0 * (n - 1)
        f = int(np.floor(virtual))
        gamma = virtual - f
        c = min(f + 1, n - 1)
        f_buf = f - (n - m)
        c_buf = c - (n - m)
        if f_buf < 0:
            raise RuntimeError(
                "streaming top-{} per channel cannot reach rank {} of {} rows for tensor "
                "{}; the calibration set grew far beyond the planned sample count".format(m, f, n, name)
            )
        # replicate numpy exactly so the streamed result is bit-identical to the
        # stacked np.percentile: quantile promotes sub-double floats to float64
        # before its _lerp (which rearranges the formula when t >= 0.5)
        a = sbuf[f_buf].astype(np.float64)
        b = sbuf[c_buf].astype(np.float64)
        diff = b - a
        if gamma >= 0.5:
            res = b - diff * (1 - gamma)
        else:
            res = a + diff * gamma
        return res.astype(np.single)


def save_stream_checkpoint(checkpoint_dir, keys, reducer, collected):
    """Atomically dump the streaming reducer's state (per tensor: top-K buffer + row
    count) plus the collected-sample count as one npz; small (K is tiny), so unlike
    the raw activations this can be flushed cheaply mid-pass. Arrays are positional
    in `keys` order (npz keys cannot safely hold every ONNX tensor-name character);
    the json metadata rides inside the npz so the whole checkpoint is one atomic file."""
    meta = {
        "names": list(keys),
        "percentile": reducer.percentile,
        "k": reducer.k,
        "collected": collected,
        "shapes": [list(reducer.state[name]["shape"]) for name in keys],
    }
    arrays = {}
    for i, name in enumerate(keys):
        st = reducer.state[name]
        arrays["topk_{}".format(i)] = st["topk"]
        arrays["n_{}".format(i)] = np.array(st["n"])
    path = pathlib.Path(checkpoint_dir) / STREAM_STATE_FILE
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def load_stream_checkpoint(checkpoint_dir, keys, reducer):
    """Restore an interrupted pass's reducer state; returns the number of samples it
    already collected (0 if absent/stale). The dumped tensor set and percentile must
    match exactly (same op_types/percentile config); on mismatch the stale state is
    deleted and 0 returned, so a changed configuration can never half-resume into
    wrong numerics."""
    path = pathlib.Path(checkpoint_dir) / STREAM_STATE_FILE
    if not path.exists():
        return 0
    try:
        with np.load(path) as z:
            meta = json.loads(z["meta"].item())
            if meta["names"] != list(keys) or meta["percentile"] != reducer.percentile:
                raise ValueError("dumped for a different tensor set or percentile")
            state = {}
            for i, name in enumerate(keys):
                state[name] = {
                    "topk": z["topk_{}".format(i)],
                    "n": int(z["n_{}".format(i)]),
                    "shape": tuple(meta["shapes"][i]),
                }
    except Exception as e:
        logger.warning("discarding stale smooth-stream checkpoint in {} ({})".format(checkpoint_dir, e))
        clear_stream_checkpoint(checkpoint_dir)
        return 0
    reducer.k = meta["k"]
    reducer.state = state
    return int(meta["collected"])


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
            checkpoint_dir (str, optional): when set, the streaming reducer's state (small:
                per-channel top-K + counts) is periodically dumped to
                <checkpoint_dir>/smooth-stream.npz and reloaded on the next run, so an
                interrupted calibration pass resumes mid-pass instead of redoing every
                forward.
            checkpoint_interval_sec (float, optional): minimum seconds between those dumps
                (0 dumps after every sample). Defaults to 1200 (20 minutes).
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

    def get_intermediate_outputs(self, checkpoint_keys=None, reducer=None):
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
            outputs = session.run(None, ort_inputs)
            if reducer is not None:
                # streaming: fold each smoothed tensor into the reducer and free the
                # sample; nothing is retained, so memory stays flat over the pass
                by_name = dict(zip(node_output_names, outputs))
                for key in checkpoint_keys:
                    reducer.add(key, by_name[key])
            else:
                for output_idx, output in enumerate(outputs):
                    output_dicts.setdefault(node_output_names[output_idx], []).append(output)

        # Mid-pass resume (streaming mode only): an interrupted run's reducer state is
        # restored and the forwards of the samples it already folded in are skipped
        # entirely. The state is tiny (per-channel top-K + counts), so the time-gated
        # dump costs little even on a slow pass; at most checkpoint_interval_sec of
        # forwards can be lost to a crash.
        checkpoint_on = bool(self.checkpoint_dir and checkpoint_keys and reducer is not None)
        restored = 0
        if checkpoint_on:
            restored = load_stream_checkpoint(self.checkpoint_dir, checkpoint_keys, reducer)
            if restored:
                logger.info(
                    "smooth-stream checkpoint: restored the state of {} collected "
                    "sample(s); skipping their forwards".format(restored)
                )
        state = {"collected": 0, "last_flush": time.time()}

        def _consume(ort_inputs):
            if state["collected"] < restored:
                state["collected"] += 1  # already folded into the restored state
                return
            _collect_data(ort_inputs)
            state["collected"] += 1
            if checkpoint_on and time.time() - state["last_flush"] >= self.checkpoint_interval_sec:
                save_stream_checkpoint(self.checkpoint_dir, checkpoint_keys, reducer, state["collected"])
                state["last_flush"] = time.time()

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
        # No tail flush: the full smooth-calib checkpoint is written immediately after
        # this pass returns and supersedes the mid-pass state anyway.
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
        # Stream the percentile instead of stacking every sample: each collected
        # sample is folded into a per-channel top-K and freed, so the pass holds
        # O(K x channels) instead of O(samples x frames x channels) (which reached
        # hundreds of GB on long-window runs). The result is bit-identical to the
        # stacked np.percentile (see StreamingChannelPercentile).
        reducer = StreamingChannelPercentile(percentile, planned_samples=len(self.iterations) or None)
        self.get_intermediate_outputs(checkpoint_keys=list(tensors_to_node.keys()), reducer=reducer)

        # remove the input tensors of {op_types} to outputs of the model
        self.model_wrapper.remove_tensors_from_outputs(tensors_to_node.keys())
        max_vals_per_channel = {}
        shape_infos = {}

        for key, val in tensors_to_node.items():
            max_vals_per_channel[key] = reducer.result(key)
            shape_infos[key] = reducer.shape(key)
            for item in val:
                shape_infos[item[1][1]] = self.model_wrapper.get_initializer(item[1][1]).dims
        return max_vals_per_channel, shape_infos, tensors_to_node
