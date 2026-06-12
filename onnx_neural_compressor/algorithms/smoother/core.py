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
"""Smoother for onnxrt."""

import copy
import io
import json
import os
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import tqdm

from onnx_neural_compressor import data_reader, logger, onnx_model
from onnx_neural_compressor.algorithms import utility as quant_utils
from onnx_neural_compressor.algorithms.smoother import calibrator

from typing import List, Union  # isort: skip


def _quiet_session_options():
    """ORT SessionOptions that silence the per-build 'Removing initializer' warnings.

    The smoother rebuilds an InferenceSession for every (node, alpha) evaluation in the
    auto-alpha search; each build logs a WARNING for every pruned ``*_smooth_scale``
    initializer (graph.cc CleanUnusedInitializersAndNodeArgs), flooding the console with
    thousands of identical lines. Severity 3 keeps errors and fatals, drops the warnings;
    the tqdm progress bar then stays the single readable signal of what is happening.
    """
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return so


def _build_qdq_session(model, providers):
    """Build the single-node QDQ sub-graph session used to score a node across the alpha grid.

    The sub-graph feeds BOTH the activation and the (alpha-dependent) quant-dequant weight
    as runtime inputs (see _make_sub_graph), so for a fixed node it is identical across every
    calibration sample AND every alpha. The session is therefore built ONCE per node and
    reused by _qdq_loss for all samples and all alphas, instead of rebuilt per (node, alpha)
    -- which was n_alphas sessions per node and made a wide alpha grid balloon native memory
    (each ort.InferenceSession arena is never handed back to the OS) until it OOM'd.
    """
    return ort.InferenceSession(
        model.SerializeToString(), sess_options=_quiet_session_options(), providers=providers
    )


def _qdq_loss(session, input_name, input_data, output_data, weight_name, weight_data):
    """Sum-of-squares loss between the fp32 reference output and the QDQ sub-graph output
    for one calibration sample, against a prebuilt session (see _build_qdq_session). Both the
    quant-dequant activation and the (alpha-dependent) quant-dequant weight are fed per call,
    so one session serves the whole alpha grid."""
    input_data = quant_utils.qdq_data(input_data, 2, False)
    preds = session.run(None, {input_name: input_data, weight_name: weight_data})
    return np.sum(np.abs(output_data - preds) ** 2)


def _format_alpha_summary(rows):
    """Render the per-layer alpha the auto-search settled on, grouped by op type.

    rows: list of (node_name, op_type, alpha). Returns a list of log lines: a per-op-type
    value histogram (how many layers picked each alpha, answering "did everything just
    land on the default, or did the search actually move layers?") followed by the full
    per-layer list so a diverging layer is identifiable by name. Pure (no logger) so it is
    unit-testable. Alphas are formatted with %g so 0.5 reads "0.5", 0.55 reads "0.55".
    """
    lines = ["auto-alpha: per-layer alpha chosen ({} smoothed node(s))".format(len(rows))]
    by_type = {}
    for name, op_type, alpha in rows:
        by_type.setdefault(op_type, []).append((name, float(alpha)))
    for op_type in sorted(by_type):
        items = by_type[op_type]
        hist = {}
        for _, alpha in items:
            hist[round(alpha, 4)] = hist.get(round(alpha, 4), 0) + 1
        summary = ", ".join("{:g}x{}".format(a, hist[a]) for a in sorted(hist))
        lines.append("  {} ({} node(s)): {}".format(op_type, len(items), summary))
        for name, alpha in items:
            lines.append("    {} -> {:g}".format(name, alpha))
    return lines


def _to_array_extern_safe(tensor, base_dir):
    """numpy array of an initializer WITHOUT materializing external bytes into the live proto.

    onnx.numpy_helper.to_array on an external-data tensor calls
    load_external_data_for_tensor, which ASSIGNS tensor.raw_data in place while leaving
    data_location EXTERNAL, so every later to_array on the same tensor assigns again.
    Under protobuf's upb backend (the default) each assignment abandons the previous bytes
    in the owning ModelProto's arena, which is only freed when the WHOLE proto dies: reads
    repeated per (node, alpha) evaluation leak weight-sized blocks for the lifetime of the
    model. Copying the (tiny, metadata-only) external tensor to a throwaway TensorProto and
    letting to_array materialize THAT keeps the bytes in the throwaway's own arena, freed on
    return. An in-proto tensor is read as-is (to_array does not mutate it).
    """
    if onnx.external_data_helper.uses_external_data(tensor):
        detached = onnx.TensorProto()
        detached.CopyFrom(tensor)
        return onnx.numpy_helper.to_array(detached, base_dir)
    return onnx.numpy_helper.to_array(tensor, base_dir)


def select_worst_nodes(losses, spec):
    """Pick the nodes whose BEST achievable QDQ loss is highest, i.e. the layers the
    alpha search could fix least; excluding them from quantization (keeping them fp32)
    is sensitivity-based mixed precision.

    losses: {node_name: normalized best loss} as recorded by _auto_tune_alpha in
    ``Smoother.auto_alpha_losses``. spec: an int n (>= 1) selects the n worst nodes,
    a float f in (0, 1) selects the worst round(f * len(losses)) nodes (at least 1).
    Ties are broken by node name so the selection is deterministic. Pure and
    unit-testable; the SmoothQuantExcludeWorst plumbing lives in smooth_quant_entry.
    """
    if isinstance(spec, bool) or not isinstance(spec, (int, float)):
        raise ValueError("exclude-worst spec must be an int >= 1 or a float in (0, 1), got {!r}".format(spec))
    if isinstance(spec, float):
        if not 0.0 < spec < 1.0:
            raise ValueError("a fractional exclude-worst spec must be in (0, 1), got {!r}".format(spec))
        count = max(1, round(len(losses) * spec))
    else:
        if spec < 1:
            raise ValueError("an integer exclude-worst spec must be >= 1, got {!r}".format(spec))
        count = min(spec, len(losses))
    ranked = sorted(losses, key=lambda name: (-losses[name], name))
    return ranked[:count]


def _atomic_write_bytes(path, data):
    """Write checkpoint bytes via a temp file + os.replace so a crash (OOM kill) mid-write
    can never leave a truncated checkpoint behind: the file either has the previous
    complete content or the new complete content."""
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, str(path))


def _smooth_calib_checkpoint_paths(checkpoint_dir):
    d = pathlib.Path(checkpoint_dir)
    return d / "smooth-calib.npz", d / "smooth-calib.json"


def load_smooth_calib_checkpoint(checkpoint_dir):
    """Load _dump_op_info's smoother-calibration results from a checkpoint dir, or None.

    Returns (max_vals_per_channel, shape_info, tensors_to_node) when both checkpoint
    files exist, else None. The npz holds the per-tensor activation-max arrays in the
    order of the json's "max_vals_names" list (names live in the json because npz keys
    cannot hold every ONNX tensor-name character safely)."""
    npz_path, json_path = _smooth_calib_checkpoint_paths(checkpoint_dir)
    if not (npz_path.exists() and json_path.exists()):
        return None
    with open(json_path) as f:
        meta = json.load(f)
    with np.load(npz_path) as npz:
        max_vals = {name: npz[f"arr_{i}"] for i, name in enumerate(meta["max_vals_names"])}
    return max_vals, meta["shape_info"], meta["tensors_to_node"]


def save_smooth_calib_checkpoint(checkpoint_dir, max_vals_per_channel, shape_info, tensors_to_node):
    """Persist _dump_op_info's results so a resumed run skips the smoother-calibration
    forward passes entirely. shape_info values and tensors_to_node node.input/output
    are protobuf repeated containers; both are converted to plain lists for json."""
    npz_path, json_path = _smooth_calib_checkpoint_paths(checkpoint_dir)
    names = list(max_vals_per_channel)
    buf = io.BytesIO()
    np.savez(buf, *[max_vals_per_channel[n] for n in names])
    _atomic_write_bytes(npz_path, buf.getvalue())
    meta = {
        "max_vals_names": names,
        "shape_info": {k: [int(d) for d in v] for k, v in shape_info.items()},
        "tensors_to_node": {
            k: [[ni[0], list(ni[1]), list(ni[2])] for ni in v] for k, v in tensors_to_node.items()
        },
    }
    _atomic_write_bytes(json_path, json.dumps(meta).encode())


def _alpha_checkpoint_path(checkpoint_dir):
    return pathlib.Path(checkpoint_dir) / "alphas.jsonl"


def load_alpha_checkpoint(checkpoint_dir):
    """Per-node auto-alpha results from a previous (possibly crashed) run, as
    {node_name: {"node", "key", "op_type", "alpha", "loss", "losses"}}. Empty when absent.

    The checkpoint is JSONL, one object per line appended as each node's grid completes
    (append-only beats rewriting the whole file per node: O(1) per node, and a crash can
    only damage the line being written). A corrupt line, i.e. the half-appended tail of
    a killed run, is skipped with a warning: that node simply gets re-searched. When a
    node somehow appears twice, the later line wins."""
    path = _alpha_checkpoint_path(checkpoint_dir)
    if not path.exists():
        return {}
    nodes = {}
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            nodes[entry["node"]] = entry
        except (ValueError, KeyError, TypeError):
            logger.warning(
                "alpha checkpoint {}: skipping corrupt line {}{}".format(
                    path, i + 1,
                    " (a truncated final append from a killed run)" if i == len(lines) - 1 else "",
                )
            )
    return nodes


def append_alpha_checkpoint(checkpoint_dir, entry):
    """Append one completed node's search result to alphas.jsonl (see
    load_alpha_checkpoint for the format and the crash-safety argument)."""
    with open(_alpha_checkpoint_path(checkpoint_dir), "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _make_sub_graph(node, inits, input_data, output_data, weight_name, weight_data, opset, ir_version):
    """Build a single-node model whose weight is a runtime INPUT, not a baked initializer.

    The QDQ-loss sub-graph is structurally identical for every alpha at a given node (same
    node, same activation/weight shapes); only the quant-dequant weight VALUES change with
    alpha. Feeding the weight as a graph input (instead of baking the alpha-dependent weight
    as an initializer) makes the built session reusable across the whole alpha grid, so it is
    built once per node rather than once per (node, alpha). bias / other non-weight
    initializers are alpha-independent and stay baked in ``inits``.

    Args:
        node (object): node
        inits (list): non-weight initializer inputs of this node (e.g. bias), kept baked
        input_data (numpy.ndarray): fp32 activation input (used for its dtype/shape)
        output_data (numpy.ndarray): fp32 output (used for its dtype/shape)
        weight_name (str): name of the weight input fed at run time (node.input[1])
        weight_data (numpy.ndarray): quant-dequant weight (used for its dtype/shape)
        opset (object): opset of the model
        ir_version (object): ir_version of the model
    """
    input = onnx.helper.make_tensor_value_info(
        node.input[0],
        onnx.helper.np_dtype_to_tensor_dtype(input_data.dtype),
        input_data.shape,
    )
    weight = onnx.helper.make_tensor_value_info(
        weight_name,
        onnx.helper.np_dtype_to_tensor_dtype(weight_data.dtype),
        weight_data.shape,
    )
    output = onnx.helper.make_tensor_value_info(
        node.output[0],
        onnx.helper.np_dtype_to_tensor_dtype(output_data.dtype),
        output_data.shape,
    )
    graph = onnx.helper.make_graph([node], "sub_graph", [input, weight], [output], inits)
    model = onnx.helper.make_model(graph, opset_imports=opset)
    model.ir_version = ir_version
    return model


class Smoother:
    """Fake input channel quantization.

    For more details please refer to:
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    We only support inplace mode which means the model weights will be changed,
    you can call recover function to recover the weights if needed.
    """

    def __init__(
        self,
        model: Union[onnx.ModelProto, onnx_model.ONNXModel, pathlib.Path, str],
        dataloader: data_reader.CalibrationDataReader,
        execution_provider: str = "CPUExecutionProvider",
    ):
        """Initialize the attributes of class."""
        self.model = (
            model if isinstance(model, onnx_model.ONNXModel) else onnx_model.ONNXModel(model, load_external_data=True)
        )
        self.value_infos = {vi.name: vi for vi in self.model.model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in self.model.model.graph.output})
        self.value_infos.update({it.name: it for it in self.model.model.graph.input})
        self.dataloader = dataloader
        self.providers = [execution_provider]
        self.tensor_scales_info = {}
        self.new_added_mul_nodes = []
        self.new_added_value_info = []
        self.new_init_tensors = []  # scales_tensor
        self.scales_per_op = True
        self.replace_input = []
        self.ops_to_absorb = []
        self.max_vals_per_channel = None
        self.shape_info = None
        self.tensors_to_node = None
        # Tensors whose per-channel layout the smoother cannot resolve (the activation
        # max axis and the weight in-channel length disagree, e.g. FastConformer's
        # relative-position attention MatMuls). They are skipped (left unsmoothed)
        # rather than crashing _get_smooth_scale on a mismatched broadcast.
        self.skipped_smooth_tensors = set()
        # Large-model auto-alpha state: one shared ORIGINAL-weight session exposing
        # every per-node tensor, plus a per-node activation cache reused across the
        # alpha grid (the activations are alpha-independent on the large path).
        self._sq_shared_session = None
        self._sq_out_cache = {"node": None, "outputs": None}
        # Per-node QDQ sub-graph session, reused across the whole alpha grid (the weight is
        # fed as a runtime input, so the session is alpha-independent). Built once per node
        # instead of once per (node, alpha) -- see _make_sub_graph / _get_output_loss.
        self._sq_subgraph = {"node": None, "session": None, "input": None, "weight": None}
        # Per-node best QDQ loss recorded by the auto-alpha search, normalized by each
        # node's reference-output energy so values are comparable across nodes. Feeds
        # select_worst_nodes / SmoothQuantExcludeWorst. Empty when alpha is a fixed
        # float (no search) or for nodes pinned to a 1-point grid (never evaluated).
        self.auto_alpha_losses = {}
        # Quiets _adjust_weights' own tqdm bar while it is called once per (node, alpha)
        # inside the auto-alpha search (where the single auto-alpha bar is the real signal);
        # the final single full-model apply in transform() re-enables it.
        self._quiet_adjust = False
        self._build_absorb_function()

    def transform(
        self,
        alpha: Union[float, str] = 0.5,
        folding: bool = True,
        percentile: float = 99.999,
        op_types: List[str] = ["Gemm", "Conv", "MatMul", "FusedConv"],
        scales_per_op: bool = True,
        calib_iter: int = 100,
        auto_alpha_args: dict = {"alpha_min": 0.3, "alpha_max": 0.7, "alpha_step": 0.05, "attn_method": "min"},
        checkpoint_dir: Union[str, pathlib.Path, None] = None,
        checkpoint_interval_sec: float = 1200,
        *args,
        **kwargs
    ):
        """The main entry of smooth quant.

        Args:
            alpha (float, optional): alpha value to balance the quantization difficulty of activation and weight.
                Defaults to 0.5.
            folding (bool, optional): whether fold those foldable Mul which are inserted for smooth quant.
                Defaults to True.
            percentile (float, optional): percentile of calibration to remove outliers.
                Defaults to 99.999.
            op_types (list, optional): the op type to be smooth quantized.
                Defaults to ["Gemm", "Conv", "MatMul", "FusedConv"].
            scales_per_op (bool, optional): True, each op will have an individual scale, mainlyfor accuracy
                False, ops with the same input will share a scale, mainly for performance.
                Defaults to True.
            calib_iter (int, optional): iteration num for calibration. Defaults to 100.
            auto_alpha_args (_type_, optional): alpha args for auto smooth.
                Defaults to {"alpha_min": 0.3, "alpha_max": 0.7, "alpha_step": 0.05, "attn_method": "min"}.
            checkpoint_dir (str | pathlib.Path, optional): directory for resumable
                intermediates. When set, the smoother-calibration results and the per-node
                auto-alpha results are written there as they are produced (smooth-calib.npz/
                .json, alphas.jsonl) and any results already present are LOADED instead of
                recomputed, so a crashed/OOM-killed run resumes from the last completed node
                instead of restarting the whole search. Caller owns cache invalidation: the
                files carry no model/config fingerprint, so a stale dir must not be reused
                across different models, calibration data, or alpha grids.
            checkpoint_interval_sec (float, optional): minimum seconds between the
                calibrator's MID-pass per-sample activation dumps (smooth-acts/); they are
                activation-sized, so a fast pass writes nothing while a slow pass loses at
                most this much work to a crash. Defaults to 1200 (20 minutes).

        Returns:
            onnx.ModelProto: A FP32 model with the same architecture as the orig model
                but with different weight which will be benefit to quantization
        """
        self.scales_per_op = scales_per_op
        self.clean()
        if isinstance(alpha, float) and (alpha < 0 or alpha > 1):
            logger.warning("alpha should be a float value in [0, 1] or 'auto' ")
            if alpha < 0:
                alpha = 0
                logger.warning("reset alpha to 0 ")
            elif alpha > 1.0:
                alpha = 1.0
                logger.warning("reset alpha to 1.0 ")

        self._dump_op_info(percentile, op_types, calib_iter, checkpoint_dir=checkpoint_dir,
                           checkpoint_interval_sec=checkpoint_interval_sec)

        if alpha == "auto":
            alpha = self._auto_tune_alpha(calib_iter, checkpoint_dir=checkpoint_dir, **auto_alpha_args)

        scales = self._get_smooth_scales(alpha)
        self._insert_smooth_mul_op(scales)
        self._adjust_weights(scales)

        self.model.add_nodes(self.new_added_mul_nodes)
        self.model.model.graph.value_info.extend(self.new_added_value_info)
        self.model.add_initializers(self.new_init_tensors)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)

        self.model.update()
        if folding:
            self._fold_scale(scales)
        self.model.topological_sort()
        self.model.remove_unused_nodes()
        return self.model.model

    def _dump_op_info(self, percentile, op_types, iterations, checkpoint_dir=None,
                      checkpoint_interval_sec=1200):
        """Dump op info for smooth quant.

        Args:
            percentile (float): percentile of calibration to remove outliers
            op_types (list): the op type to be smooth quantized
            iterations (int): iterations
            checkpoint_dir (str, optional): when set, load the calibration results from
                there if present (skipping every forward pass), else compute and save them.
                The calibrator additionally dumps its per-sample activations in there
                (time-gated) so even a run killed MID-pass resumes from the last dumped
                sample instead of redoing every forward.
            checkpoint_interval_sec (float): minimum seconds between those mid-pass dumps
        """
        loaded = load_smooth_calib_checkpoint(checkpoint_dir) if checkpoint_dir else None
        if loaded is not None:
            self.max_vals_per_channel, self.shape_info, self.tensors_to_node = loaded
            logger.info(
                "smooth-calib checkpoint: loaded {} activation-max tensor(s) from {}; "
                "skipping the smoother calibration forwards".format(
                    len(self.max_vals_per_channel), checkpoint_dir
                )
            )
            # Partial per-sample dumps from the run that crashed before finishing the
            # full checkpoint are superseded by it now; drop the dead weight.
            calibrator.clear_acts_checkpoint(checkpoint_dir)
        else:
            sq_calibrator = calibrator.Calibrator(
                self.model,
                self.dataloader,
                iterations=list(range(0, iterations)),
                # NOTE: the ctor parameter is `providers`; the historical call passed
                # `execution_provider=`, which fell into **kwargs and silently pinned
                # every smoother-calibration forward to the CPU even on a CUDA run.
                providers=self.providers,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_sec=checkpoint_interval_sec,
            )

            self.max_vals_per_channel, self.shape_info, self.tensors_to_node = sq_calibrator.calib_smooth(
                op_types, percentile
            )
            if checkpoint_dir:
                save_smooth_calib_checkpoint(
                    checkpoint_dir, self.max_vals_per_channel, self.shape_info, self.tensors_to_node
                )
                calibrator.clear_acts_checkpoint(checkpoint_dir)
        for node in self.model.nodes():
            for out in node.output:
                if (
                    out in self.tensors_to_node
                    and node.op_type in self.could_absorb_optype
                    and self.model.get_initializer(node.input[1]) is not None
                ):
                    self.ops_to_absorb.append(node.name)

    def recover(self):
        """Recover the model weights."""
        for tensor_name, nodes in self.tensors_to_node.items():
            for node_info in nodes:
                key = node_info[0] if self.scales_per_op else tensor_name
                if key not in self.tensor_scales_info:
                    continue
                input = node_info[1][1]
                weight = _to_array_extern_safe(
                    self.model.get_initializer(input),
                    base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
                )
                scale = self.tensor_scales_info[key]
                new_weight = weight * scale
                self.model.set_initializer(input, new_weight)

        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, new_input_name, old_input_name)

        for value_info in self.new_added_value_info:
            self.model.model.graph.value_info.remove(value_info)

        self.model.remove_nodes(self.new_added_mul_nodes)
        self.model.remove_initializers(self.new_init_tensors)
        self.tensor_scales_info = {}
        self.new_added_mul_nodes = []
        self.new_init_tensors = []
        self.new_added_value_info = []
        self.replace_input = []

    def clean(self):
        """Clean data collected from calibration."""
        self.tensor_scales_info = {}
        self.new_added_mul_nodes = []
        self.new_init_tensors = []
        self.new_added_value_info = []
        self.replace_input = []

    def _build_absorb_function(self):
        """Build function mapping for scale folding."""

        def norm(node, scale):  # pragma: no cover
            for idx in [1, 2]:
                tensor = self.model.get_initializer(node.input[idx])
                new_tensor = (
                    onnx.numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale
                    if self.model.model_path is not None
                    else onnx.numpy_helper.to_array(tensor) * scale
                )
                self.model.set_initializer(node.input[idx], new_tensor)
                self.tensor_scales_info[node.input[idx]] = (
                    1.0 / scale
                    if node.input[idx] not in self.tensor_scales_info
                    else self.tensor_scales_info[node.input[idx]] * 1.0 / scale
                )
            return True

        def mul(node, scale):  # pragma: no cover
            if all([self.model.get_initializer(inp) is None for inp in node.input]):
                return False
            for inp in node.input:
                if self.model.get_initializer(inp) is not None:
                    key = node.input[0].split("_smooth_output")[0]
                    tensor = self.model.get_initializer(inp)
                    new_tensor = (
                        onnx.numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale
                        if self.model.model_path is not None
                        else onnx.numpy_helper.to_array(tensor) * scale
                    )
                    # set_initializer requires the dims of old & new initializers are same
                    # Mul operator has broadcast mechanism
                    self.model.remove_initializer(tensor)
                    self.model.add_initializer(
                        onnx.helper.make_tensor(
                            inp, tensor.data_type, list(new_tensor.shape), new_tensor.flatten().tolist()
                        )
                    )
                    self.tensor_scales_info[key] = (
                        1.0 / scale
                        if key not in self.tensor_scales_info
                        else 1.0 / scale * self.tensor_scales_info[key]
                    )
            return True

        def conv(node, scale):  # pragma: no cover
            if len(node.input) > 2:
                if self.model.get_initializer(node.input[2]) is not None:
                    tensor = self.model.get_initializer(node.input[2])
                    new_tensor = (
                        onnx.numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale
                        if self.model.model_path is not None
                        else onnx.numpy_helper.to_array(tensor) * scale
                    )
                    self.model.set_initializer(node.input[2], new_tensor)
                    self.tensor_scales_info[node.input[2]] = 1.0 / scale
                scale = scale.reshape(-1, 1, 1, 1)
                tensor = self.model.get_initializer(node.input[1])
                new_tensor = (
                    onnx.numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale
                    if self.model.model_path is not None
                    else onnx.numpy_helper.to_array(tensor) * scale
                )
                self.model.set_initializer(node.input[1], new_tensor)
                self.tensor_scales_info[node.input[1]] = (
                    1.0 / scale
                    if node.input[1] not in self.tensor_scales_info
                    else self.tensor_scales_info[node.input[1]] * 1.0 / scale
                )
            return True

        self.could_absorb_optype = {
            "LayerNormalization": norm,
            "BatchNormalization": norm,
            "InstanceNormalization": norm,
            "SimplifiedLayerNormalization": mul,
            "MatMul": mul,
            "Gemm": mul,
            "Conv": conv,
            "FusedConv": conv,
            "Mul": mul,
        }

    def _fold_scale(self, scales):
        """Absorb the scale to the operator at output channel.

        Args:
            scales (dict): scales for smooth quant, {tensor_name: smooth quant scale}
        """
        remove_nodes = []
        for node in self.model.nodes():
            if node.op_type == "Mul" and node.name.endswith("_smooth_mul") and node not in remove_nodes:
                parent = self.model.get_parent(node, 0)
                if parent is None:
                    continue
                if parent.op_type in self.could_absorb_optype and len(self.model.get_children(parent)) == 1:
                    if node.output[0].split("_smooth_output")[0] in scales:
                        if self.could_absorb_optype[parent.op_type](
                            parent, 1.0 / scales[node.output[0].split("_smooth_output")[0]]
                        ):
                            remove_nodes.append(node)
                            children = [i for i in self.model.nodes() if node.output[0] in i.input]
                            for child in children:
                                for idx, inp in enumerate(child.input):
                                    if inp == node.output[0]:
                                        child.input[idx] = node.input[0]
        self.model.remove_nodes(remove_nodes)

    def _reference_activations(self, node, node_name):
        """Return the per-sample (node-input, node-output) reference activations the QDQ loss
        scores against, as a list of (np.ndarray, np.ndarray).

        Small model: recomputed every call from a fresh full-model session (the activations
        depend on the just-smoothed weights, so they are not cacheable across the alpha grid).
        Large model: harvested ONCE per node from the shared ORIGINAL-weight augment session
        (alpha-independent) and cached across the whole grid.

        The calibration ``dataloader`` is rewound before every harvest: the alpha search
        drains it once in ``_dump_op_info`` and never rewinds before searching, so without
        this each evaluation would see an exhausted reader (loss 0) and the per-layer optimal
        alpha would collapse to ``alpha_min`` for every node.
        """
        if not self.model.is_large_model:
            orig_outputs = self.model.output()
            added_tensors = [node.input[0], node.output[0]]
            self.model.add_tensors_to_outputs(added_tensors)
            session = ort.InferenceSession(
                self.model.model.SerializeToString(),
                sess_options=_quiet_session_options(),
                providers=self.providers,
            )
            self.model.remove_tensors_from_outputs([i for i in added_tensors if i not in orig_outputs])
            self.dataloader.rewind()
            samples = []
            while True:
                inputs = self.dataloader.get_next()
                if not inputs:
                    break
                out0, out1 = session.run(added_tensors, inputs)
                samples.append((out0, out1))
            return samples

        # Large model: the augment file (saved once in _auto_tune_alpha with the ORIGINAL
        # weights and every per-node in/out tensor exposed as a graph output) backs ONE
        # shared session whose harvested activations are alpha-independent, so cache them
        # per node across the whole alpha grid.
        if self._sq_out_cache.get("node") != node_name:
            if self._sq_shared_session is None:
                self._sq_shared_session = ort.InferenceSession(
                    self.model.model_path + "_augment.onnx",
                    sess_options=_quiet_session_options(),
                    providers=self.providers,
                )
            fetch = [node.input[0], node.output[0]]
            self.dataloader.rewind()
            collected = []
            while True:
                inputs = self.dataloader.get_next()
                if not inputs:
                    break
                out0, out1 = self._sq_shared_session.run(fetch, inputs)
                collected.append((out0, out1))
            self._sq_out_cache = {"node": node_name, "outputs": collected}
        return self._sq_out_cache["outputs"]

    def _get_output_loss(self, node_name, scale, calib_iter, weight=None):
        """Get output loss of specific node after inserting QDQ pair.

        Args:
            node_name (str): node name
            scale (float): scale of the specific node
            calib_iter (int): iterations
            weight (np.ndarray, optional): the candidate (already smooth-scaled) weight to
                score. When given, the node's initializer is not read at all: the large-model
                auto-alpha search passes the scaled weight directly so the search never has
                to write it into the proto first (see _scaled_weight on why those writes leak).

        The (alpha-dependent) quant-dequant weight is the ONLY thing that changes across the
        alpha grid for a fixed node, so it is fed to the sub-graph session as a runtime input
        (see _make_sub_graph) and the session itself is built once per node and cached on
        ``self._sq_subgraph``. This keeps the count of ort.InferenceSession builds at one per
        node instead of one per (node, alpha): a wide alpha grid no longer multiplies session
        churn (whose never-reclaimed native arenas were OOM-ing wide-grid searches).
        """
        node = [i for i in self.model.nodes() if i.name == node_name]
        loss = 0
        if len(node) == 0:
            return loss
        node = node[0]
        base_dir = os.path.dirname(self.model.model_path) if self.model.model_path is not None else ""

        # The quant-dequant weight is alpha-dependent (the weight was just scaled for this
        # alpha, either in-proto by _adjust_weights or passed in directly); it is fed to the
        # cached session per call. bias / other non-weight initializers are alpha-independent
        # and stay baked into the sub-graph.
        if weight is None:
            weight = _to_array_extern_safe(self.model.get_initializer(node.input[1]), base_dir)
        weight_q = quant_utils.qdq_data(weight, 3, True)
        extra_inits = [
            self.model.get_initializer(i)
            for i in node.input[2:]
            if self.model.get_initializer(i) is not None
        ]

        samples = self._reference_activations(node, node_name)
        if not samples:
            return loss

        if self._sq_subgraph.get("node") != node_name:
            ref_in, ref_out = samples[0]
            sub_model = _make_sub_graph(
                node, extra_inits, ref_in, ref_out, node.input[1], weight_q,
                self.model.model.opset_import, self.model.model.ir_version,
            )
            self._sq_subgraph = {
                "node": node_name,
                "session": _build_qdq_session(sub_model, self.providers),
                "input": sub_model.graph.input[0].name,
                "weight": node.input[1],
                # Reference-output energy of this node, the normalizer that makes
                # per-node best losses comparable across nodes (the raw sum-of-squares
                # loss scales with each node's output magnitude). Computed once per
                # node alongside the session; on the small path the references drift
                # slightly with the in-proto weight of the alpha under evaluation,
                # which is fine for a ranking.
                "ref_sq": float(sum((s[1].astype(np.float64) ** 2).sum() for s in samples)),
            }
        sub_sess = self._sq_subgraph["session"]
        sub_input = self._sq_subgraph["input"]
        weight_name = self._sq_subgraph["weight"]

        for ref_in, ref_out in samples:
            loss += _qdq_loss(sub_sess, sub_input, ref_in * scale, ref_out, weight_name, weight_q)
        return loss

    def _reshape_scale_for_input(self, tensor, key):
        """Reshape the scale for input feature in channel.

        Args:
            tensor (str): tensor name
            key (str): scale key of this tensor
        """
        if len(self.shape_info[tensor]) == 4:
            scale = np.reshape(self.tensor_scales_info[key], (1, self.tensor_scales_info[key].shape[1], 1, 1))
        else:
            scale = np.reshape(self.tensor_scales_info[key], (1, self.tensor_scales_info[key].shape[0]))
        return scale

    def _alpha_grid(self, alpha_min, alpha_max, alpha_step):
        """Inclusive-of-both-endpoints alpha search grid.

        A request to search alpha_min..alpha_max should try alpha_max itself (np.arange
        stops just short of its stop value). The +alpha_step/2 nudge pulls alpha_max inside
        the half-open range without risking a spurious extra step from floating-point
        rounding. A degenerate grid (alpha_min == alpha_max) collapses to a single value.
        """
        return np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step).tolist()

    def _node_alpha_space(self, node, default_space, op_alpha, alpha_min, alpha_max, alpha_step):
        """The alpha grid to search for one node, honouring per-op-type overrides.

        op_alpha maps an op type to either a fixed float (pin the alpha, no search: a
        single-point grid) or a dict of {alpha_min, alpha_max, alpha_step} (a custom grid,
        each key defaulting to the global value). An op type absent from op_alpha uses the
        global default_space.
        """
        override = (op_alpha or {}).get(node.op_type)
        if override is None:
            return default_space
        if isinstance(override, dict):
            return self._alpha_grid(
                override.get("alpha_min", alpha_min),
                override.get("alpha_max", alpha_max),
                override.get("alpha_step", alpha_step),
            )
        return [float(override)]  # pinned: a single-point grid, evaluated for free below

    def _auto_tune_alpha(
        self,
        calib_iter,
        alpha_min: float = 0.3,
        alpha_max: float = 0.7,
        alpha_step: float = 0.05,
        attn_method: str = "min",
        op_alpha: dict = None,
        checkpoint_dir=None,
    ):
        """Perform alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly.

        Args:
            calib_iter (int): iterations
            alpha_min (float): min value of alpha search space.
            alpha_max (float): max value of alpha search space.
            alpha_step (float): step size of alpha search space.
            attn_method (str): criterion method used on attention ops; currently min, max and mean are supported.
            op_alpha (dict): optional per-op-type alpha override, {op_type: float | {alpha_min, alpha_max, alpha_step}}.
                A float pins that op type's alpha (no search, one forward pass saved per node);
                a dict gives it its own search grid; op types absent here use the global grid.
            checkpoint_dir (str, optional): when set, one line is appended to alphas.jsonl
                there after EVERY node's grid finishes (best alpha, normalized best loss,
                and the full per-alpha loss curve), and nodes already recorded in it are
                restored instead of re-searched. This is what lets an OOM-killed multi-hour
                search resume from the last completed node. The caller is responsible for
                not reusing the dir across different models/calibration data/grids.
        """
        logger.info("auto tuning alpha")

        default_space = self._alpha_grid(alpha_min, alpha_max, alpha_step)

        optimal_alphas = {}
        self.auto_alpha_losses = {}

        # Warn for op_alpha keys that match no smoothed node: setting an alpha for an op
        # type that is not in op_types, or that has no weight to migrate outliers into
        # (e.g. Slice), is a silent no-op otherwise.
        smoothed_types = {
            self.model.get_node(ni[0]).op_type
            for infos in self.tensors_to_node.values()
            for ni in infos
        }
        for op_type in (op_alpha or {}):
            if op_type not in smoothed_types:
                logger.warning(
                    "op_alpha set for {!r} but no smoothed node has that op type "
                    "(not in op_types, or it has no weight to smooth); ignored".format(op_type)
                )

        # Per-node alpha grid (a pinned op collapses to a 1-point grid). The (node, alpha)
        # QDQ-loss evaluation is the unit of work and by far the slowest phase, so drive a
        # tqdm bar over it (live count, rate and ETA). _adjust_weights is called once per
        # evaluation here, so silence its own bar for the duration.
        node_spaces = {
            ni[0]: self._node_alpha_space(
                self.model.get_node(ni[0]), default_space, op_alpha, alpha_min, alpha_max, alpha_step
            )
            for infos in self.tensors_to_node.values()
            for ni in infos
        }
        n_nodes = len(node_spaces)
        total = sum(len(s) for s in node_spaces.values())

        # Resume state: nodes whose grid already completed in a previous run of the SAME
        # search (the caller keys checkpoint_dir by its full configuration). An entry only
        # counts when its recorded scales key matches what THIS search would use, so a
        # checkpoint from a different scales_per_op mode can never be silently misread.
        ckpt_nodes = load_alpha_checkpoint(checkpoint_dir) if checkpoint_dir else {}
        cached = {
            ni[0]
            for tname, infos in self.tensors_to_node.items()
            for ni in infos
            if ckpt_nodes.get(ni[0], {}).get("key") == (ni[0] if self.scales_per_op else tname)
        }
        if ckpt_nodes:
            logger.info(
                "auto-alpha checkpoint: {} of {} node(s) already searched in {}; "
                "resuming with the remaining {}".format(
                    len(cached), n_nodes, checkpoint_dir, n_nodes - len(cached)
                )
            )

        logger.info(
            "auto-alpha: {} node(s), {} QDQ-loss evaluation(s) over the per-node alpha grids".format(
                n_nodes, total
            )
        )
        self._quiet_adjust = True
        pbar = tqdm.tqdm(total=total, desc="SmoothQuant auto-alpha", unit="eval", leave=True)

        # Large-model path: expose every per-node in/out tensor as a graph output
        # BEFORE saving the augment file, so a single shared original-weight session
        # can fetch them (the stock augment file lacks them and session.run raises
        # "Invalid output name"). Those activations are alpha-independent, which is what
        # makes the per-node cache in _get_output_loss valid.
        self._sq_shared_session = None
        self._sq_out_cache = {"node": None, "outputs": None}
        self._sq_subgraph = {"node": None, "session": None, "input": None, "weight": None}
        added_outputs = []
        # When every multi-point node is already checkpointed (or pinned), no QDQ loss is
        # ever evaluated, so skip the expensive augment save (a full external-data copy of
        # the model) entirely; the end-of-search reload/cleanup is skipped to match.
        needs_eval = any(len(space) > 1 and name not in cached for name, space in node_spaces.items())
        augment_saved = False
        if self.model.is_large_model and needs_eval:
            augment_saved = True
            needed, seen = [], set()
            for node_infos in self.tensors_to_node.values():
                for node_info in node_infos:
                    n = self.model.get_node(node_info[0])
                    for t in (n.input[0], n.output[0]):
                        if t and t not in seen:
                            seen.add(t)
                            needed.append(t)
            added_outputs = [t for t in needed if t not in set(self.model.output())]
            self.model.add_tensors_to_outputs(added_outputs)
            onnx.save_model(
                self.model.model,
                self.model.model_path + "_augment.onnx",
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.pb",
                convert_attribute=False,
            )

        ## Searching optimal alphas
        try:
            for tensor_name, node_infos in self.tensors_to_node.items():
                for node_info in node_infos:
                    loss_alpha = {}
                    key = node_info[0] if self.scales_per_op else tensor_name
                    node = self.model.get_node(node_info[0])
                    space = node_spaces[node_info[0]]
                    if node_info[0] in cached:
                        # Checkpointed by a previous run of this same search: restore the
                        # best alpha and (for searched nodes) the normalized best loss the
                        # exclude-worst ranking needs, and skip the whole grid.
                        entry = ckpt_nodes[node_info[0]]
                        optimal_alphas[key] = entry["alpha"]
                        if entry.get("loss") is not None:
                            self.auto_alpha_losses[node_info[0]] = entry["loss"]
                        pbar.update(len(space))
                        continue
                    if len(space) == 1:
                        # Pinned alpha (or a degenerate grid): nothing to compare, so record
                        # it and skip the QDQ loss evaluation entirely. This is the whole cost
                        # saving of pinning an op you already trust (e.g. MatMul=0.5).
                        optimal_alphas[key] = space[0]
                        if checkpoint_dir:
                            append_alpha_checkpoint(checkpoint_dir, {
                                "node": node_info[0], "key": key, "op_type": node.op_type,
                                "alpha": space[0], "loss": None, "losses": {}, "pinned": True,
                            })
                        pbar.update(1)
                        continue
                    for alpha in space:
                        scale = self._get_smooth_scales(alpha, [key])
                        if self.model.is_large_model:
                            # Score the candidate WITHOUT writing the proto. The historical
                            # flow (_adjust_weights then recover) rewrote this node's weight
                            # initializer twice per evaluation; under protobuf's upb backend
                            # each rewrite permanently grows the ModelProto's arena (old
                            # bytes are only freed when the whole proto dies), so the
                            # retained memory scaled with n_nodes x n_alphas and survived
                            # into static calibration: a 0.1-step grid OOMed where a
                            # 0.2-step grid fit. The references are alpha-independent here
                            # (original-weight augment session), so nothing else needs the
                            # adjusted weight: compute it in numpy and feed it directly.
                            # Bonus: each alpha now scales the PRISTINE on-disk weight
                            # instead of one that drifted through k scale/unscale roundtrips.
                            cand_weight, inv_scale = self._scaled_weight(node_info[1][1], key, scale)
                            self.tensor_scales_info[key] = inv_scale
                        else:
                            # Small path: the reference activations are recomputed per call
                            # from the live model, so the weight must really be adjusted
                            # in-proto (and recovered below) to keep the loss semantics.
                            cand_weight = None
                            self._adjust_weights(scale)
                        input_scale = (
                            self._reshape_scale_for_input(tensor_name, key)
                            if not (node.op_type == "Gemm" and quant_utils.is_B_transposed(node))
                            else self.tensor_scales_info[key]
                        )
                        loss = self._get_output_loss(node_info[0], input_scale, calib_iter, weight=cand_weight)
                        loss_alpha[alpha] = loss
                        if key not in optimal_alphas:  # Update alpha results
                            optimal_alphas[key] = alpha
                        else:
                            optimal_alphas[key] = (
                                alpha
                                if optimal_alphas[key] in loss_alpha and loss < loss_alpha[optimal_alphas[key]]
                                else optimal_alphas[key]
                            )
                        if self.model.is_large_model:
                            self.tensor_scales_info = {}
                        else:
                            self.recover()
                        pbar.update(1)
                    if loss_alpha:
                        # The node's best achievable loss, normalized by its reference-output
                        # energy (cached on _sq_subgraph, still holding THIS node right after
                        # its grid): the sensitivity ranking behind SmoothQuantExcludeWorst.
                        ref_sq = (
                            self._sq_subgraph.get("ref_sq")
                            if self._sq_subgraph.get("node") == node_info[0]
                            else None
                        )
                        self.auto_alpha_losses[node_info[0]] = min(loss_alpha.values()) / (ref_sq or 1.0)
                    if checkpoint_dir:
                        # Checkpoint after EVERY completed node grid so a crash/OOM loses at
                        # most one node's work. "losses" keeps the full raw per-alpha curve
                        # (the search criterion), "loss" the normalized best (the ranking).
                        append_alpha_checkpoint(checkpoint_dir, {
                            "node": node_info[0],
                            "key": key,
                            "op_type": node.op_type,
                            "alpha": optimal_alphas[key],
                            "loss": self.auto_alpha_losses.get(node_info[0]),
                            "losses": {"{:g}".format(a): float(l) for a, l in loss_alpha.items()},
                        })
        finally:
            pbar.close()
            self._quiet_adjust = False
            if self.model.is_large_model:
                self.model.remove_tensors_from_outputs(added_outputs)
            self._sq_shared_session = None
            self._sq_out_cache = {"node": None, "outputs": None}
            self._sq_subgraph = {"node": None, "session": None, "input": None, "weight": None}

        logger.info("auto tuning alpha done")
        self._log_alpha_summary(optimal_alphas)
        if self.model.is_large_model and augment_saved:

            onnx.external_data_helper.load_external_data_for_model(
                self.model.model, os.path.split(self.model.model_path)[0]
            )
            os.remove(self.model.model_path + "_augment.onnx")
            os.remove(os.path.join(os.path.dirname(self.model.model_path), "weights.pb"))
        return optimal_alphas

    def _log_alpha_summary(self, optimal_alphas):
        """Log which alpha the search actually picked for every smoothed layer.

        Tells you at a glance whether the default just won everywhere or whether the
        per-layer search genuinely moved layers. optimal_alphas is keyed by node name
        when scales_per_op, else by the shared input tensor feeding several nodes; both
        are expanded to one (node, op_type, alpha) row so the report is always per layer.
        """
        rows = []
        for tensor_name, node_infos in self.tensors_to_node.items():
            for node_info in node_infos:
                key = node_info[0] if self.scales_per_op else tensor_name
                if key not in optimal_alphas:
                    continue
                node = self.model.get_node(node_info[0])
                rows.append((node_info[0], node.op_type, optimal_alphas[key]))
        if not rows:
            return
        for line in _format_alpha_summary(rows):
            logger.info(line)

    def _get_smooth_scales(self, alpha, target_list=[]):
        """Get the smooth scales for.

        The ops with the same input will share one mul layer.
        TODO support individual scales for each layer.

        Args:
            alpha: smooth alpha in paper
            target_list: target objects to get scale, [] means get all scales

        Returns:
            the smooth scales for weights, currently one input tensor only have one scale
        """
        scales = {}
        for tensor, nodes in self.tensors_to_node.items():
            # if scales_per_op the key of scales is the node name, otherwise the activation of node
            if self.scales_per_op:
                for node_info in nodes:
                    node = self.model.get_node_by_weight(node_info[1][1])
                    if len(target_list) > 0 and node_info[0] not in target_list:
                        continue
                    weight = _to_array_extern_safe(
                        self.model.get_initializer(node_info[1][1]),
                        base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
                    )
                    if (len(weight.shape) == 4 and weight.shape[1] != 1) or (
                        node.op_type == "Gemm" and quant_utils.is_B_transposed(node)
                    ):
                        weight = np.moveaxis(weight, 0, 1)
                    specific_alpha = alpha[node_info[0]] if isinstance(alpha, dict) else alpha
                    scales[node_info[0]] = self._get_smooth_scale(weight, specific_alpha, tensor)
            else:
                if len(target_list) > 0 and tensor not in target_list:
                    continue
                weights_in_channel_max = []
                for node_info in nodes:
                    node = self.model.get_node_by_weight(node_info[1][1])
                    weight = _to_array_extern_safe(
                        self.model.get_initializer(node_info[1][1]),
                        base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
                    )
                    if (len(weight.shape) == 4 and weight.shape[1] != 1) or (
                        node.op_type == "Gemm" and quant_utils.is_B_transposed(node)
                    ):
                        weight = np.moveaxis(weight, 0, 1)
                    weight = weight.reshape(weight.shape[0], -1)
                    cur_max = np.amax(weight, axis=-1)
                    weights_in_channel_max.append(cur_max)
                weights_stack = np.stack(weights_in_channel_max, axis=-1)
                specific_alpha = alpha[tensor] if isinstance(alpha, dict) else alpha
                scales[tensor] = self._get_smooth_scale(weights_stack, specific_alpha, tensor)

        # Drop nodes _get_smooth_scale could not resolve (returned None); downstream
        # consumers (_insert_smooth_mul_op / _adjust_weights) iterate scales.keys().
        scales = {k: v for k, v in scales.items() if v is not None}
        if not target_list and self.skipped_smooth_tensors:
            logger.info(
                "SmoothQuant left {} node(s) unsmoothed (unresolvable per-channel layout); "
                "they fall through to plain static quantization".format(len(self.skipped_smooth_tensors))
            )
        return scales

    def _get_smooth_scale(self, weights, specific_alpha, tensor):
        """Get smooth scale for specific weight.

        Args:
            weights (numpy.ndarray): weight data
            specific_alpha (float): current alpha for this weights
            tensor (str): tensor name
        """
        weights = np.abs(weights.reshape(weights.shape[0], -1))
        weights_max = np.amax(weights, axis=-1)
        if self.max_vals_per_channel[tensor].shape != weights_max.shape:
            # The per-channel activation max and the weight in-channel length disagree
            # (the smoother assumes a 3D activation is (batch, seq, in_channel) with the
            # in-channel LAST, which does not hold for e.g. FastConformer's relative-
            # position attention MatMuls). The smooth scale could not be broadcast, so
            # skip this node (it falls through to plain static quantization) instead of
            # crashing. _insert_smooth_mul_op / _adjust_weights both guard on the key,
            # so a missing scale is safe.
            self.skipped_smooth_tensors.add(tensor)
            return None
        input_power = np.power(self.max_vals_per_channel[tensor], specific_alpha)
        weight_power = np.power(weights_max, 1 - specific_alpha)
        weight_power = np.clip(weight_power, a_min=1e-5, a_max=None)
        scale = np.clip(input_power / weight_power, a_min=1e-5, a_max=None)
        return scale

    def _insert_smooth_mul_op(self, scales):
        """Insert the Mul after inupt.

        The ops with the same input will share one mul layer.

        Args:
            scales (dict): The smooth scales
        """
        for key in scales.keys():
            input_name = key if not self.scales_per_op else self.model.get_node(key).input[0]
            weight_name = (
                self.tensors_to_node[key][0][1][1] if not self.scales_per_op else self.model.get_node(key).input[1]
            )
            scale_factor = 1.0 / scales[key]
            if (
                len(self.shape_info[weight_name]) == 3 or len(self.shape_info[weight_name]) == 2
            ):  # the last dim is input channel
                pass
            elif len(self.shape_info[weight_name]) == 4:
                scale_factor = np.reshape(scale_factor, (1, -1, 1, 1))
            else:
                assert False, "not support"
            name = key + "_" + "smooth_scale"
            scale_tensor = onnx.helper.make_tensor(
                name=key + "_" + "smooth_scale",
                data_type=onnx.TensorProto.FLOAT,
                dims=scale_factor.shape,
                vals=scale_factor.flatten().tolist(),
            )
            self.new_init_tensors.append(scale_tensor)
            mul_output_name = key + "_smooth_output"
            mul_node = onnx.helper.make_node(
                "Mul",
                inputs=[input_name, key + "_" + "smooth_scale"],
                outputs=[mul_output_name],
                name=key + "_smooth_mul",
            )
            self.new_added_mul_nodes.append(mul_node)
            if input_name in self.value_infos:
                value_info = copy.deepcopy(self.value_infos[input_name])
                value_info.name = mul_node.output[0]
                self.new_added_value_info.append(value_info)
            if self.scales_per_op:
                self.replace_input.append([self.model.get_node(key), input_name, mul_output_name])
            else:
                for node_info in self.tensors_to_node[key]:
                    self.replace_input.append([self.model.get_node(node_info[0]), key, mul_output_name])

    def _scaled_weight(self, input_name, key, scales):
        """Scaled candidate weight for one node, computed WITHOUT writing the model proto.

        Returns (new_weight, inv_scale): the weight with the smooth scale folded in, plus
        the reshaped inverse scale that belongs in ``tensor_scales_info[key]`` (what
        _reshape_scale_for_input and recover() consume). _adjust_weights delegates here
        for its per-node math so the two can never drift apart; the auto-alpha search
        calls it directly to score an alpha without mutating any initializer (under
        protobuf's upb backend, rewriting an initializer of a long-lived ModelProto
        abandons the old bytes in the proto's arena until the WHOLE model dies, so
        per-evaluation rewrites leak memory proportional to the alpha-grid size).
        """
        node = self.model.get_node_by_weight(input_name)
        weight = _to_array_extern_safe(
            self.model.get_initializer(input_name),
            base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
        )
        if len(weight.shape) == 2:
            scale = (
                np.expand_dims(scales[key], axis=0)
                if node.op_type == "Gemm" and quant_utils.is_B_transposed(node)
                else np.expand_dims(scales[key], axis=-1)
            )
        elif len(weight.shape) == 4:  # TODO need to check conv
            if (
                weight.shape[1] == 1
                and "group" in [i.name for i in node.attribute]
                and [i for i in node.attribute if i.name == "group"][0].i > 1
            ):
                scale = np.reshape(scales[key], (-1, 1, 1, 1))
            else:
                scale = np.reshape(scales[key], (1, -1, 1, 1))
        else:
            assert False, "not support"
        return weight * scale, 1.0 / scale

    def _adjust_weights(self, scales):
        """Adjust the weights with scale.

        Args:
            scales (dict): The input scales
        """
        for tensor_name, nodes in tqdm.tqdm(
            self.tensors_to_node.items(),
            desc="SmoothQuant: folding scales into weights",
            unit="tensor",
            disable=self._quiet_adjust,
            leave=False,
        ):
            for node_info in nodes:
                key = node_info[0] if self.scales_per_op else tensor_name
                if key not in scales:
                    continue
                input = node_info[1][1]
                new_weight, inv_scale = self._scaled_weight(input, key, scales)
                self.tensor_scales_info[key] = inv_scale

                new_tensor = onnx.numpy_helper.from_array(new_weight, input)
                self.model.get_initializer(input).CopyFrom(new_tensor)
