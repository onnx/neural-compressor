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


def _get_quant_dequant_output(model, input_data, output_data, providers):
    """Get loss between fp32 output and QDQ output.

    Args:
        model (object): model
        input_data (numpy.ndarray): fp32 input
        output_data (numpy.ndarray): fp32 output
        providers (list): execution provider
    """
    input_data = quant_utils.qdq_data(input_data, 2, False)
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=_quiet_session_options(), providers=providers
    )
    preds = sess.run(None, {model.graph.input[0].name: input_data})
    loss = np.sum(np.abs(output_data - preds) ** 2)
    return loss


def _make_sub_graph(node, inits, input_data, output_data, opset, ir_version):
    """Build a model with the specific node.

    Args:
        node (object): node
        inits (list): initializer inputs of this node
        input_data (numpy.ndarray): fp32 input
        output_data (numpy.ndarray): fp32 output
        opset (object): opset of the model
        ir_version (object): ir_version of the model
    """
    input = onnx.helper.make_tensor_value_info(
        node.input[0],
        onnx.helper.np_dtype_to_tensor_dtype(input_data.dtype),
        input_data.shape,
    )
    output = onnx.helper.make_tensor_value_info(
        node.output[0],
        onnx.helper.np_dtype_to_tensor_dtype(output_data.dtype),
        output_data.shape,
    )
    graph = onnx.helper.make_graph([node], "sub_graph", [input], [output], inits)
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

        self._dump_op_info(percentile, op_types, calib_iter)

        if alpha == "auto":
            alpha = self._auto_tune_alpha(calib_iter, **auto_alpha_args)

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

    def _dump_op_info(self, percentile, op_types, iterations):
        """Dump op info for smooth quant.

        Args:
            percentile (float): percentile of calibration to remove outliers
            op_types (list): the op type to be smooth quantized
            iterations (int): iterations
        """
        sq_calibrator = calibrator.Calibrator(
            self.model,
            self.dataloader,
            iterations=list(range(0, iterations)),
            execution_provider=self.providers,
        )

        self.max_vals_per_channel, self.shape_info, self.tensors_to_node = sq_calibrator.calib_smooth(
            op_types, percentile
        )
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
                weight = onnx.numpy_helper.to_array(
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

    def _get_output_loss(self, node_name, scale, calib_iter):
        """Get output loss of specific node after inserting QDQ pair.

        Args:
            node_name (str): node name
            scale (float): scale of the specific node
            calib_iter (int): iterations

        The calibration ``dataloader`` is rewound before every evaluation: the alpha
        search drains it once in ``_dump_op_info`` and never rewinds before searching,
        so without this each evaluation would see an exhausted reader (loss 0) and the
        per-layer optimal alpha would collapse to ``alpha_min`` for every node.
        """
        node = [i for i in self.model.nodes() if i.name == node_name]
        loss = 0
        if len(node) == 0:
            return loss
        node = node[0]

        if not self.model.is_large_model:
            # Small model: the session is cheap to rebuild and must be rebuilt every
            # call (the weight was just scaled for this alpha), so keep doing that and
            # only add the missing rewind.
            orig_outputs = self.model.output()
            added_tensors = [node.input[0], node.output[0]]
            self.model.add_tensors_to_outputs(added_tensors)
            session = ort.InferenceSession(
                self.model.model.SerializeToString(),
                sess_options=_quiet_session_options(),
                providers=self.providers,
            )
            weight = onnx.numpy_helper.to_array(self.model.get_initializer(node.input[1]), "")
            weight_q = quant_utils.qdq_data(weight, 3, True)
            self.model.set_initializer(node.input[1], weight_q)
            inits = [self.model.get_initializer(i) for i in node.input if self.model.get_initializer(i) is not None]

            self.dataloader.rewind()
            model = None
            while True:
                inputs = self.dataloader.get_next()
                if not inputs:
                    break
                outputs = session.run(added_tensors, inputs)
                if model is None:
                    model = _make_sub_graph(
                        node, inits, outputs[0], outputs[1],
                        self.model.model.opset_import, self.model.model.ir_version,
                    )
                loss += _get_quant_dequant_output(model, outputs[0] * scale, outputs[1], self.providers)
            self.model.remove_tensors_from_outputs([i for i in added_tensors if i not in orig_outputs])
            self.model.set_initializer(node.input[1], weight)
            return loss

        # Large model: the augment file (saved once in _auto_tune_alpha with the
        # ORIGINAL weights and every per-node in/out tensor exposed as a graph output)
        # backs ONE shared session whose harvested activations are alpha-independent.
        # Harvest each node's activations once and cache them across the whole alpha
        # grid; only the cheap per-alpha QDQ sub-graph depends on the scaled weight.
        base_dir = os.path.dirname(self.model.model_path)
        weight = onnx.numpy_helper.to_array(self.model.get_initializer(node.input[1]), base_dir)
        weight_q = quant_utils.qdq_data(weight, 3, True)
        self.model.set_initializer(node.input[1], weight_q)
        inits = [self.model.get_initializer(i) for i in node.input if self.model.get_initializer(i) is not None]

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

        model = None
        for out0, out1 in self._sq_out_cache["outputs"]:
            if model is None:
                model = _make_sub_graph(
                    node, inits, out0, out1,
                    self.model.model.opset_import, self.model.model.ir_version,
                )
            loss += _get_quant_dequant_output(model, out0 * scale, out1, self.providers)
        self.model.set_initializer(node.input[1], weight)
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
        """
        logger.info("auto tuning alpha")

        default_space = self._alpha_grid(alpha_min, alpha_max, alpha_step)

        optimal_alphas = {}

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
        added_outputs = []
        if self.model.is_large_model:
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
                    if len(space) == 1:
                        # Pinned alpha (or a degenerate grid): nothing to compare, so record
                        # it and skip the QDQ loss evaluation entirely. This is the whole cost
                        # saving of pinning an op you already trust (e.g. MatMul=0.5).
                        optimal_alphas[key] = space[0]
                        pbar.update(1)
                        continue
                    for alpha in space:
                        scale = self._get_smooth_scales(alpha, [key])
                        self._adjust_weights(scale)
                        input_scale = (
                            self._reshape_scale_for_input(tensor_name, key)
                            if not (node.op_type == "Gemm" and quant_utils.is_B_transposed(node))
                            else self.tensor_scales_info[key]
                        )
                        loss = self._get_output_loss(node_info[0], input_scale, calib_iter)
                        loss_alpha[alpha] = loss
                        if key not in optimal_alphas:  # Update alpha results
                            optimal_alphas[key] = alpha
                        else:
                            optimal_alphas[key] = (
                                alpha
                                if optimal_alphas[key] in loss_alpha and loss < loss_alpha[optimal_alphas[key]]
                                else optimal_alphas[key]
                            )
                        self.recover()
                        pbar.update(1)
        finally:
            pbar.close()
            self._quiet_adjust = False
            if self.model.is_large_model:
                self.model.remove_tensors_from_outputs(added_outputs)
            self._sq_shared_session = None
            self._sq_out_cache = {"node": None, "outputs": None}

        logger.info("auto tuning alpha done")
        if self.model.is_large_model:

            onnx.external_data_helper.load_external_data_for_model(
                self.model.model, os.path.split(self.model.model_path)[0]
            )
            os.remove(self.model.model_path + "_augment.onnx")
            os.remove(os.path.join(os.path.dirname(self.model.model_path), "weights.pb"))
        return optimal_alphas

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
                    weight = onnx.numpy_helper.to_array(
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
                    weight = onnx.numpy_helper.to_array(
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
                node = self.model.get_node_by_weight(input)
                weight = onnx.numpy_helper.to_array(
                    self.model.get_initializer(input),
                    base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
                )
                if len(weight.shape) == 2:
                    scale = (
                        np.expand_dims(scales[key], axis=0)
                        if node.op_type == "Gemm" and quant_utils.is_B_transposed(node)
                        else np.expand_dims(scales[key], axis=-1)
                    )
                    new_weight = weight * scale
                elif len(weight.shape) == 4:  # TODO need to check conv
                    node = self.model.get_node_by_weight(input)
                    if (
                        weight.shape[1] == 1
                        and "group" in [i.name for i in node.attribute]
                        and [i for i in node.attribute if i.name == "group"][0].i > 1
                    ):
                        scale = np.reshape(scales[key], (-1, 1, 1, 1))
                    else:
                        scale = np.reshape(scales[key], (1, -1, 1, 1))
                    new_weight = weight * scale
                else:
                    assert False, "not support"
                self.tensor_scales_info[key] = 1.0 / scale

                new_tensor = onnx.numpy_helper.from_array(new_weight, input)
                self.model.get_initializer(input).CopyFrom(new_tensor)
