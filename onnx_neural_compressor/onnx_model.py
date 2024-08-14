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
"""Class for ONNX model."""

import collections
import copy
import os
import pathlib
import sys

import onnx
import transformers

from onnx_neural_compressor import constants, logger, utility


class ONNXModel:
    """Build ONNX model."""

    def __init__(self, model, **kwargs):
        """Initialize an ONNX model.

        Args:
            model (str or ModelProto): path to onnx model or loaded ModelProto model object.
        """
        self.model = model if not isinstance(model, str) else onnx.load(model, load_external_data=False)
        self._model_path = None if not isinstance(model, str) else model
        self.check_is_large_model()
        if self._is_large_model and self._model_path is None and not kwargs.get("ignore_warning", False):
            logger.warning("Model size > 2GB. Please use model path instead of onnx model object to quantize")

        if self._is_large_model and isinstance(model, str) and kwargs.get("load_external_data", True):
            onnx.external_data_helper.load_external_data_for_model(self.model, os.path.dirname(self._model_path))

        self._config = None
        if isinstance(model, str) and os.path.exists(pathlib.Path(model).parent.joinpath("config.json").as_posix()):
            self._config = transformers.PretrainedConfig.from_pretrained(pathlib.Path(model).parent.as_posix())
        self.node_name_counter = {}
        self._output_name_to_node = {}
        self._input_name_to_nodes = {}
        self._get_output_name_to_node(self.model.graph.node)
        self._get_input_name_to_nodes(self.model.graph.node)
        self._graph_info = {}
        self._get_graph_info()
        self._q_config = None

    def output_name_to_node(self):
        self._output_name_to_node = {}
        self._get_output_name_to_node(self.model.graph.node)
        return self._output_name_to_node

    def input_name_to_nodes(self):
        self._input_name_to_nodes = {}
        self._get_input_name_to_nodes(self.model.graph.node)
        return self._input_name_to_nodes

    def _get_input_name_to_nodes(self, nodes):
        """Get input names of nodes."""
        for node in nodes:
            attrs = [
                attr
                for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
            ]
            if len(attrs) > 0:
                for attr in attrs:
                    self._get_input_name_to_nodes(attr.g.node)
            for input_name in node.input:
                if len(input_name.strip()) != 0:
                    if input_name not in self._input_name_to_nodes:
                        self._input_name_to_nodes[input_name] = [node]
                    else:
                        self._input_name_to_nodes[input_name].append(node)

    def _get_output_name_to_node(self, nodes):
        """Get output names of nodes."""
        for node in nodes:
            attrs = [
                attr
                for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
            ]
            if len(attrs) > 0:
                for attr in attrs:
                    self._get_output_name_to_node(attr.g.node)
            for output_name in node.output:
                if len(output_name.strip()) != 0:
                    self._output_name_to_node[output_name] = node

    @property
    def model_path(self):
        """Return model path."""
        return self._model_path

    @model_path.setter
    def model_path(self, path):
        """Set model path."""
        self._model_path = path

    def check_is_large_model(self):
        """Check model > 2GB."""
        init_size = 0
        for init in self.model.graph.initializer:
            # if initializer has external data location, return True
            if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
                self._is_large_model = True
                return
            # if raise error of initializer size > 2GB, return True
            try:
                init_bytes = init.SerializeToString()
                init_size += sys.getsizeof(init_bytes)
            except Exception as e:
                if "exceeds maximum protobuf size of 2GB" in str(e):
                    self._is_large_model = True
                    return
                else:  # pragma: no cover
                    raise e
            if init_size > constants.MAXIMUM_PROTOBUF:
                self._is_large_model = True
                return
        self._is_large_model = False

    @property
    def is_large_model(self):
        """Check the onnx model is over 2GB."""
        return self._is_large_model

    @property
    def framework(self):
        """Return framework."""
        return "onnxruntime"

    def add_initializer(self, tensor):
        """Add a initializer to model."""
        if tensor.name not in [i.name for i in self._model.graph.initializer]:
            self._model.graph.initializer.append(tensor)

    def add_initializers(self, tensors):
        """Add initializers to model."""
        for tensor in tensors:
            self.add_initializer(tensor)

    @property
    def q_config(self):
        """Return q_config."""
        return self._q_config

    @q_config.setter
    def q_config(self, q_config):
        """Set q_config."""
        self._q_config = q_config

    @property
    def hf_config(self):
        """Return huggingface config if model is Transformer-based."""
        return self._config

    def input(self):
        """Return input of model."""
        return [i.name for i in self.model.graph.input]

    def output(self):
        """Return output of model."""
        return [i.name for i in self.model.graph.output]

    @property
    def model(self):
        """Return model itself."""
        return self._model

    @model.setter
    def model(self, model):
        """Set model itself."""
        self._model = model
        self._graph_info = {}
        self._get_graph_info()
        self._output_name_to_node = {}
        self._input_name_to_nodes = {}
        self._get_input_name_to_nodes(self._model.graph.node)
        self._get_output_name_to_node(self._model.graph.node)

    def nodes(self):
        """Return model nodes."""
        return self._model.graph.node

    def initializer(self):
        """Return model initializer."""
        return self._model.graph.initializer

    def graph(self):
        """Return model graph."""
        return self._model.graph

    @property
    def ir_version(self):
        """Return model ir_version."""
        return self._model.ir_version

    @property
    def opset_import(self):
        """Return model opset_import."""
        return self._model.opset_import

    def update(self):
        """Update model info."""
        self._graph_info = {}
        self._get_graph_info()
        self._output_name_to_node = self.output_name_to_node()
        self._input_name_to_nodes = self.input_name_to_nodes()

    @property
    def graph_info(self):
        """Return ORT Graph Info object holding information about backend graph."""
        return self._graph_info

    def _get_graph_info(self):
        """Update graph info."""
        for node in self.model.graph.node:
            self.graph_info.update({node.name: node.op_type})

    def is_graph_output(self, name):
        """Check whether the tensor is the graph output."""
        return name in self.output()

    def save(self, root):
        """Save ONNX model."""
        if os.path.split(root)[0] != "" and not os.path.exists(os.path.split(root)[0]):
            os.mkdir(os.path.split(root)[0])
        if self.is_large_model:  # pragma: no cover
            onnx.external_data_helper.load_external_data_for_model(self.model, os.path.split(self._model_path)[0])
            onnx.save_model(
                self.model,
                root,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=os.path.basename(root) + "_data",
                size_threshold=1024,
                convert_attribute=False,
            )
        else:
            onnx.save(self.model, root)

        self._model_path = root

        if self._config is not None and not os.path.exists(os.path.join(os.path.split(root)[0], "config.json")):
            model_type = "" if not hasattr(self._config, "model_type") else getattr(self._config, "model_type")
            setattr(self._config.__class__, "model_type", model_type)
            output_config_file = pathlib.Path(root).parent.joinpath("config.json").as_posix()
            self._config.to_json_file(output_config_file, use_diff=False)

    def remove_initializer(self, tensor):
        """Remove an initializer from model."""
        if tensor in self._model.graph.initializer:
            self._model.graph.initializer.remove(tensor)

    def remove_initializers(self, init_to_remove):
        """Remove initializers from model."""
        for initializer in init_to_remove:
            self.remove_initializer(initializer)

    def get_initializer(self, name):
        """ "Find the initializer with specified name."""
        for initializer in self.model.graph.initializer:
            if initializer.name == name:
                return initializer
        return None

    def remove_node(self, node):
        """Remove a node from model."""
        if node in self._model.graph.node:
            self._model.graph.node.remove(node)

    def remove_nodes(self, nodes_to_remove):
        """Remove nodes from model."""
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node):
        """Add a node to model."""
        self._model.graph.node.extend([node])

    def add_nodes(self, nodes_to_add):
        """Add nodes to model."""
        self._model.graph.node.extend(nodes_to_add)

    def get_children(self, node, input_name_to_nodes=None):
        """Get children nodes."""
        if input_name_to_nodes is None:
            input_name_to_nodes = self._input_name_to_nodes

        children = []
        for output in node.output:
            if output in input_name_to_nodes:
                for child in input_name_to_nodes[output]:
                    children.append(child)
        return children

    def get_initializer_share_num(self, name):
        """Get the number of shares of initializer."""
        num = 0
        if self.get_initializer(name) is None:
            return num

        for node in self.nodes():
            if name in node.input:
                num += 1
        return num

    def get_node(self, name):
        """Get a node by name."""
        for node in self.model.graph.node:
            if node.name == name:
                return node
        return None

    def get_parent(self, node, idx, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self._output_name_to_node
        if len(node.input) <= idx:
            return None

        input = node.input[idx]
        return output_name_to_node.get(input, None)

    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self._output_name_to_node

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents

    def get_node_by_weight(self, weight_name):
        """Get a node by its weight name."""
        if len(self._input_name_to_nodes) == 0:
            self._input_name_to_nodes = self.input_name_to_nodes()
        nodes = self._input_name_to_nodes[weight_name]
        if len(nodes) == 1:
            return nodes[0]
        elif len(nodes) == 0:
            raise ValueError("{} is not used by any node in this model.".format(weight_name))
        else:
            raise NotImplementedError("Models with shared weights is not supported.")

    def set_initializer(self, tensor, array, raw=False):
        """Update initializer."""
        old_tensor = self.get_initializer(tensor)
        self.remove_initializer(old_tensor)
        dims = old_tensor.dims
        data_type = old_tensor.data_type
        new_tensor = (
            onnx.helper.make_tensor(tensor, data_type, dims, array.flatten().tolist())
            if not raw
            else onnx.helper.make_tensor(tensor, data_type, dims, array.tostring(), raw=raw)
        )
        self.add_initializer(new_tensor)

    def get_siblings(self, node):
        """Get siblings nodes."""
        siblings = []
        for parent in self.get_parents(node):
            for child in self.get_children(parent):
                if child.name != node.name:
                    siblings.append(child)
        return siblings

    def get_scale_zero(self, tensor):
        """Help function to get scale and zero_point."""
        if not tensor.endswith("_quantized"):
            logger.debug("Find {} in the quantized graph is not quantized.".format(tensor))
            return None, None

        if len(self._input_name_to_nodes) == 0:
            self._input_name_to_nodes = self.input_name_to_nodes()
        if len(self._output_name_to_node) == 0:
            self._output_name_to_node = self.output_name_to_node()

        def _searcher(tensor_name):
            """Search scale and zero point tensor recursively."""
            node = self._input_name_to_nodes[tensor_name][0]
            parent = self._output_name_to_node[tensor_name] if tensor_name in self._output_name_to_node else None
            direct_int8 = ["Reshape", "Transpose", "Squeeze", "Unsqueeze", "MaxPool", "Pad", "Split"]
            if parent is not None and parent.op_type in direct_int8:
                fp32_tensor_name = (
                    parent.input[0]
                    .replace("_quantized", "")
                    .replace("_QuantizeLinear", "")
                    .replace("_QuantizeInput", "")
                )
            elif node.op_type in ["Gather"]:  # pragma: no cover
                fp32_tensor_name = (
                    node.output[0]
                    .replace("_quantized", "")
                    .replace("_QuantizeLinear", "")
                    .replace("_QuantizeInput", "")
                )
            else:
                fp32_tensor_name = (
                    tensor_name.replace("_quantized", "").replace("_QuantizeLinear", "").replace("_QuantizeInput", "")
                )
            scale = fp32_tensor_name + "_scale"
            scale_tensor = self.get_initializer(scale)
            zo = fp32_tensor_name + "_zero_point"
            zo_tensor = self.get_initializer(zo)

            if scale_tensor is None or zo_tensor is None:
                if parent is not None:
                    scale_tensor, zo_tensor = _searcher(parent.input[0])
            return scale_tensor, zo_tensor

        node = self._input_name_to_nodes[tensor][0]
        # TODO check if scale_tensor and zero_point is needed
        # for bias of qlinearconv, scale and zero_point is not needed
        if (node.op_type == "QLinearConv" and tensor == node.input[-1]) or (
            node.op_type == "QGemm" and tensor == node.input[-3]
        ):
            return None, None
        else:
            scale_tensor, zo_tensor = _searcher(tensor)
            assert scale_tensor, "missing scale for tensor {}".format(tensor)
            assert zo_tensor, "missing zero point for tensor {}".format(tensor)
            return scale_tensor, zo_tensor

    @staticmethod
    def replace_node_input(node, old_input_name, new_input_name):
        """Replace input of a node."""
        assert isinstance(old_input_name, str) and isinstance(new_input_name, str)
        for j in range(len(node.input)):
            if node.input[j] == old_input_name:
                node.input[j] = new_input_name

    @staticmethod
    def replace_node_output(node, old_output_name, new_output_name):
        """Replace output of a node."""
        assert isinstance(old_output_name, str) and isinstance(new_output_name, str)
        for j in range(len(node.output)):
            if node.output[j] == old_output_name:
                node.output[j] = new_output_name

    def replace_input_of_all_nodes(self, old_input_name, new_input_name, white_optype=[], black_optype=[]):
        """Replace inputs of all nodes."""
        if len(white_optype) > 0:
            for node in self.model.graph.node:
                if node.op_type in white_optype:
                    ONNXModel.replace_node_input(node, old_input_name, new_input_name)
        else:
            for node in self.model.graph.node:
                if node.op_type not in black_optype:
                    ONNXModel.replace_node_input(node, old_input_name, new_input_name)

    def replace_output_of_all_nodes(self, old_output_name, new_output_name, white_optype=[], black_optype=[]):
        """Replace outputs of all nodes."""
        if len(white_optype) > 0:
            for node in self.model.graph.node:
                if node.op_type in white_optype:
                    ONNXModel.replace_node_output(node, old_output_name, new_output_name)
        else:
            for node in self.model.graph.node:
                if node.op_type not in black_optype:
                    ONNXModel.replace_node_output(node, old_output_name, new_output_name)

    def remove_duplicate_nodes(self):
        """remove duplicate nodes"""
        new_nodes = []
        for node in self.nodes():
            if node not in new_nodes:
                new_nodes.append(node)
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(new_nodes)
        self.update()

    def remove_unused_nodes(self):
        """Remove unused nodes."""
        unused_nodes = []
        for node in self.model.graph.node:
            # remove constant
            if node.op_type == "Constant":
                tensor = node.attribute[0].t
                tensor.name = node.output[0]
                self.add_initializer(tensor)
                unused_nodes.append(node)

            # remove identity
            if node.op_type == "Identity":
                tensor = self.get_initializer(node.input[0])
                if tensor is not None:
                    new_tensor = copy.deepcopy(tensor)
                    new_tensor.name = node.output[0]
                    unused_nodes.append(node)
                    self.add_initializer(new_tensor)
        self.remove_nodes(unused_nodes)

        if len(self._input_name_to_nodes) == 0:
            self._input_name_to_nodes = self.input_name_to_nodes()
        if len(self._output_name_to_node) == 0:
            self._output_name_to_node = self.output_name_to_node()

        unvalid_nodes = [
            i
            for i in self.model.graph.node
            if all(out not in self._input_name_to_nodes and out not in self.output() for out in i.output)
        ]
        while len(unvalid_nodes) > 0:
            self.remove_nodes(unvalid_nodes)
            self._input_name_to_nodes = self.input_name_to_nodes()
            unvalid_nodes = [
                i
                for i in self.model.graph.node
                if all([out not in self._input_name_to_nodes and out not in self.output() for out in i.output])
            ]

        ununsed_weights = []
        for w in self.model.graph.initializer:
            if w.name not in self._input_name_to_nodes and w.name not in self.output():
                ununsed_weights.append(w)
                # Remove from graph.input
                for graph_input in self.graph().input:
                    if graph_input.name == w.name:
                        self.graph().input.remove(graph_input)

        self.remove_initializers(ununsed_weights)
        self.update()
        self.topological_sort()

    def topological_sort(self, enable_subgraph=False):
        """Topological sort the model."""
        if not enable_subgraph:
            input_name_to_nodes = {}
            output_name_to_node = {}
            for node in self.model.graph.node:
                for input_name in node.input:
                    if len(input_name.strip()) != 0:
                        if input_name not in input_name_to_nodes:
                            input_name_to_nodes[input_name] = [node]
                        else:
                            input_name_to_nodes[input_name].append(node)
                for output_name in node.output:
                    if len(output_name.strip()) != 0:
                        output_name_to_node[output_name] = node
        else:  # pragma: no cover
            if len(self._input_name_to_nodes) == 0:
                self._input_name_to_nodes = self.input_name_to_nodes()
            if len(self._output_name_to_node) == 0:
                self._output_name_to_node = self.output_name_to_node()
            input_name_to_nodes = self._input_name_to_nodes
            output_name_to_node = self._output_name_to_node

        all_nodes = {}
        q = collections.deque()
        wait = collections.deque()
        for inp in self.model.graph.input:
            q.extend(input_name_to_nodes[inp.name])
        for n in self.model.graph.node:
            if all([i not in output_name_to_node and i not in self.input() for i in n.input]):
                q.append(n)

        while q:
            n = q.popleft()
            if not all([output_name_to_node[i].name in all_nodes for i in n.input if i in output_name_to_node]):
                if n not in wait:
                    wait.append(n)
                continue

            all_nodes[n.name] = n
            for out in n.output:
                if out in input_name_to_nodes:
                    q.extend([i for i in input_name_to_nodes[out] if i.name not in all_nodes and i not in q])
            if len(q) == 0 and len(wait) != 0:
                q = copy.deepcopy(wait)
                wait.clear()
        nodes = [i[1] for i in all_nodes.items()]
        assert len(list(set([n.name for n in nodes]))) == len(list(set([n.name for n in self.model.graph.node])))
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(nodes)

    def add_tensors_to_outputs(self, tensor_names):
        """Add the tensors to the model outputs to gets their values.

        Args:
            tensor_names: The names of tensors to be dumped.
        """
        added_outputs = []
        for tensor in tensor_names:
            if tensor not in self.output():
                added_tensor = onnx.helper.ValueInfoProto()
                added_tensor.name = tensor
                added_outputs.append(added_tensor)
        self.model.graph.output.extend(added_outputs)  # pylint: disable=no-member

    def remove_tensors_from_outputs(self, tensor_names):
        """Remove the tensors from the model outputs.

        Args:
            tensor_names: The names of tensors to be removed.
        """
        removed_outputs = []
        for tensor in tensor_names:
            if tensor in self.output():
                removed_outputs.append(self.model.graph.output[self.output().index(tensor)])
        for output in removed_outputs:
            self.model.graph.output.remove(output)

    def match_first_parent(self, node, parent_op_type, output_name_to_node_dict, exclude=[]):
        """Find parent node based on constraints on op_type.

        Args:
            node (str): current node name.
            parent_op_type (str): constraint of parent node op_type.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            exclude (list): list of nodes that are excluded (not allowed to match as parent).

        Returns:
            parent: The matched parent node. None if not found.
            index: The input index of matched parent node. None if not found.
        """
        for i, input in enumerate(node.input):
            if input in output_name_to_node_dict:
                parent = output_name_to_node_dict[input]
                if parent.op_type == parent_op_type and parent not in exclude:
                    return parent, i
        return None, None

    def match_parent(
        self,
        node,
        parent_op_type,
        input_index=None,
        output_name_to_node_dict=None,
        exclude=[],
        return_indice=None,
    ):
        """Find parent node based on constraints on op_type and index.

        Args:
            node (str): current node name.
            parent_op_type (str): constraint of parent node op_type.
            input_index (int or None): only check the parent given input index of current node.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            exclude (list): list of nodes that are excluded (not allowed to match as parent).
            return_indice (list): a list to append the input index when input_index is None.

        Returns:
            parent: The matched parent node.
        """
        assert node is not None
        assert input_index is None or input_index >= 0

        if output_name_to_node_dict is None:
            if len(self._output_name_to_node) == 0:
                self._output_name_to_node = self.output_name_to_node()
            output_name_to_node_dict = self._output_name_to_node

        if input_index is None:
            parent, index = self.match_first_parent(node, parent_op_type, output_name_to_node_dict, exclude)
            if return_indice is not None:
                return_indice.append(index)
            return parent

        if input_index >= len(node.input):
            return None

        parent = self.get_parent(node, input_index, output_name_to_node_dict)
        if parent is not None and parent.op_type == parent_op_type and parent not in exclude:
            return parent

        return None

    def match_parent_path(
        self,
        node,
        parent_op_types,
        parent_input_index,
        output_name_to_node_dict=None,
        return_indice=None,
    ):
        """Find a sequence of input edges based on constraints on parent op_type and index.

        Args:
            node (str): current node name.
            parent_op_types (str): constraint of parent node op_type of each input edge.
            parent_input_index (list): constraint of input index of each input edge.
                                       None means no constraint.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            return_indice (list): a list to append the input index when there is
                                  no constraint on input index of an edge.

        Returns:
            parents: a list of matched parent node.
        """
        assert len(parent_input_index) == len(parent_op_types)

        if output_name_to_node_dict is None:
            if len(self._output_name_to_node) == 0:
                self._output_name_to_node = self.output_name_to_node()
            output_name_to_node_dict = self._output_name_to_node

        current_node = node
        matched_parents = []
        for i, op_type in enumerate(parent_op_types):
            matched_parent = self.match_parent(
                current_node,
                op_type,
                parent_input_index[i],
                output_name_to_node_dict,
                exclude=[],
                return_indice=return_indice,
            )
            if matched_parent is None:
                return None

            matched_parents.append(matched_parent)
            current_node = matched_parent

        return matched_parents

    def is_smoothquant_model(self):
        """Check the model is smooth quantized or not.

        Returns:
            bool: the model is smooth quantized or not.
        """
        for init in self.model.graph.initializer:
            if "_smooth_scale" in init.name:
                return True
        return False

    # below functions are used for layer-wise
    def find_split_node_for_layer_wise_quantization(self):
        """Find split node for layer wise quantization."""
        # find split nodes of decoder blocks
        # embed -> decoder.0 -(split_node)-> ... -(split_node)-> decoder.n -(split_node)-> norm -> head
        # after split: embed -> decoder.0,
        #              decoder.1,
        #              decoder.2,
        #              ...,
        #              decoder.n,
        #              norm -> head
        start_nodes = []
        for node in self.model.graph.node:
            start_node, qkv_nodes_list = None, None
            if node.op_type == "SkipLayerNormalization":
                start_node = node
                qkv_nodes_list = [
                    self.match_parent_path(
                        start_node,
                        ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
                        [None, 0, 0, 0, 0],
                    ),
                    self.match_parent_path(
                        start_node,
                        ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
                        [1, 1, 0, 0, 0],
                    ),
                ]
            if node.op_type == "Add":
                start_node = node
                qkv_nodes_list = [
                    # match base attention structure
                    self.match_parent_path(
                        start_node,
                        ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
                        [0, None, 0, 0, 0],
                    ),
                    self.match_parent_path(
                        start_node, ["Add", "MatMul", "Reshape", "Transpose", "MatMul"], [1, None, 0, 0, 0]
                    ),
                    # match gpt attention no past structure
                    self.match_parent_path(
                        start_node,
                        ["Reshape", "Gemm", "Reshape", "Reshape", "Transpose", "MatMul"],
                        [None, 0, 0, 0, 0, 0],
                        output_name_to_node_dict=self._output_name_to_node,
                        return_indice=[],
                    ),
                    # match bart attention structure
                    self.match_parent_path(
                        start_node,
                        ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
                        [0, None, 0, 0, 0, 0],
                    ),
                    self.match_parent_path(
                        start_node,
                        ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
                        [1, None, 0, 0, 0, 0],
                    ),
                    self.match_parent_path(
                        start_node,
                        ["MatMul", "Mul", "MatMul", "Mul", "Div", "Add"],
                        [None, 0, None, 0, None, 0],
                    ),
                    self.match_parent_path(
                        start_node,
                        ["MatMul", "Mul", "MatMul", "SimplifiedLayerNormalization", "Add"],
                        [None, 0, None, 0, 0],
                    ),
                ]
            if qkv_nodes_list is not None and any(qkv_nodes_list):
                start_nodes.append(start_node)

        # can't find qkv nodes with above patterns, use Softmax nodes to split model
        if len(start_nodes) == 0:
            for node in self.model.graph.node:
                if node.op_type == "Softmax":
                    start_nodes.append(node)
        return start_nodes

    def find_split_nodes(self):
        """Find split nodes for layer-wise quantization."""
        self.remove_unused_nodes()
        split_nodes = self.find_split_node_for_layer_wise_quantization()
        return split_nodes

    def _infer_tensor_dtype(self):
        """Infer the elem_type of tensors."""
        initializers = dict([(i.name, i.data_type) for i in self.model.graph.initializer])
        inputs = dict([(i.name, i.type.tensor_type.elem_type) for i in self.model.graph.input])
        value_info = dict([(i.name, i.type.tensor_type.elem_type) for i in self.model.graph.value_info])
        outputs = dict([(i.name, i.type.tensor_type.elem_type) for i in self.model.graph.output])
        for node in self.model.graph.node:
            if node.output[0] in value_info:
                continue
            elem_type = None
            if node.op_type in ["And", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Or", "Xor"]:
                elem_type = onnx.TensorProto.BOOL
            elif node.op_type in ["ArgMax", "ArgMin", "NonZero", "Shape"]:
                elem_type = onnx.TensorProto.INT64
            elif node.op_type == "Cast" and len(node.attribute) > 0:
                elem_type = node.attribute[0].i
            elif node.op_type in ["Constant", "ConstantOfShape"] and len(node.attribute) > 0:
                elem_type = node.attribute[0].t.data_type
            elif len(node.input) >= 2:
                for inp in node.input[:2]:
                    if inp in initializers and initializers[inp] != onnx.TensorProto.INT64:
                        elem_type = initializers[inp]
                        break

            # output elem_type aligns with input
            if elem_type is None and len(node.input) > 0:
                inp = node.input[0]
                if inp in value_info:
                    elem_type = value_info[inp]
                elif inp in inputs:
                    elem_type = inputs[inp]
                elif inp in outputs:
                    elem_type = outputs[inp]
            if elem_type is not None:
                if node.op_type in ["Split", "Slice"]:
                    for out in node.output:
                        value_info.update({out: elem_type})
                else:
                    value_info.update({node.output[0]: elem_type})

        return value_info

    def _build_input_output_tensor(self, tensor_name, value_info):
        if tensor_name in self.input():
            return self.model.graph.input[self.input().index(tensor_name)]
        if tensor_name in self.output():
            return self.model.graph.output[self.output().index(tensor_name)]
        tensor_type = value_info.get(tensor_name, onnx.TensorProto.FLOAT)
        return onnx.helper.make_tensor_value_info(tensor_name, tensor_type, None)

    def split_model_with_node(
        self, split_node_name, path_of_model_to_split, save_both_split_models=True, save_path=None
    ):
        """Split model into two parts at a given node.

        Args:
            split_node_name (str): name of the node where the model is split at>
            path_of_model_to_split (str): path of model to be split.
            save_both_split_models (bool): whether to save the two split models.
                False means only save the first split model.
                True means save both the two split models.
                Default id True.
            save_path (str): path to save split models. None means using self.model_path

        Returns:
            tuple: the first split model, the second split model
        """
        # origin model : ... -> node_1 -> split_node -> node_2 -> ...
        # split model 1: ... -> node_1 -> split_node
        # split model 2: node_2 -> ...

        # infer elem_type of tensors to make sure layer-wise quant run successfully
        value_info = self._infer_tensor_dtype()

        split_model_part_1 = onnx.ModelProto()
        split_model_part_1.CopyFrom(self.model)
        split_model_part_1.graph.ClearField("node")

        split_model_part_2 = onnx.ModelProto()
        split_model_part_2.CopyFrom(self.model)
        split_model_part_2.graph.ClearField("node")

        split_node = None
        nodes = []
        for node in self.model.graph.node:
            nodes.append(node)

            if node.name == split_node_name:
                split_node = node
                break

        assert len(split_node.output) == 1, (
            "Only support split at node with 1 output tensor, while "
            "current split node {} has {} output tensors".format(split_node_name, len(split_node.output))
        )
        split_tensor_name = split_node.output[0]

        split_tensor = self._build_input_output_tensor(split_tensor_name, value_info)

        split_model_part_1.graph.node.extend(nodes)
        split_model_part_1.graph.output.append(split_tensor)
        split_model_part_1 = ONNXModel(split_model_part_1, ignore_warning=True)

        # remove isolated graphs which are not related to the split_node
        output_name_to_node = split_model_part_1.output_name_to_node()
        valid_nodes = [split_node]
        while len(valid_nodes) > 0:
            node = valid_nodes.pop(0)
            for inp in node.input:
                if inp in output_name_to_node:
                    valid_nodes.append(output_name_to_node[inp])
            if node in nodes:
                nodes.remove(node)
        split_model_part_1.remove_nodes(nodes)

        for node in self.model.graph.node:
            if node not in split_model_part_1.nodes():
                split_model_part_2.graph.node.append(node)

        split_model_part_2.graph.input.append(split_tensor)
        split_model_part_2 = ONNXModel(split_model_part_2, ignore_warning=True)

        # remove unused input & output
        split_model_part_1._remove_unused_input_output()
        split_model_part_2._remove_unused_input_output()

        insert_output_for_model_1 = []
        insert_input_for_model_2 = []
        for output in split_model_part_1._output_name_to_node.keys():
            if output in split_model_part_2._input_name_to_nodes.keys():
                output_tensor = self._build_input_output_tensor(output, value_info)
                if output_tensor not in split_model_part_1.model.graph.output:
                    insert_output_for_model_1.append(output_tensor)
                if output_tensor not in split_model_part_2.model.graph.input:
                    insert_input_for_model_2.append(output_tensor)

        # insert model 1 output
        for output in insert_output_for_model_1:
            split_model_part_1.model.graph.output.append(output)

        # insert model 2 input
        for input in insert_input_for_model_2:
            split_model_part_2.model.graph.input.append(input)

        # remove unused init
        split_model_part_1.remove_unused_init()
        split_model_part_2.remove_unused_init()

        split_model_part_1.update()
        split_model_part_2.update()

        dir_of_model_to_split = os.path.dirname(path_of_model_to_split)

        split_model_part_1.load_model_initializer_by_tensor(dir_of_model_to_split)
        split_model_part_1_path = (
            os.path.join(save_path, "split_model_part_1.onnx")
            if save_path is not None
            else os.path.join(dir_of_model_to_split, "split_model_part_1.onnx")
        )
        split_model_part_1.model_path = split_model_part_1_path
        split_model_part_1._save_split_model(split_model_part_1_path)
        split_model_part_1.check_is_large_model()
        logger.debug("save split model part 1 to {} for layer wise quantization".format(split_model_part_1_path))

        if save_both_split_models:
            split_model_part_2.load_model_initializer_by_tensor(dir_of_model_to_split)
            split_model_part_2_path = (
                os.path.join(save_path, "split_model_part_2.onnx")
                if save_path is not None
                else os.path.join(dir_of_model_to_split, "split_model_part_2.onnx")
            )
            split_model_part_2.model_path = split_model_part_2_path
            split_model_part_2._save_split_model(split_model_part_2_path)
            split_model_part_2.check_is_large_model()
            logger.debug("save split model part 2 to {} for layer wise quantization".format(split_model_part_2_path))
            return split_model_part_1, split_model_part_2
        else:
            return split_model_part_1, split_model_part_2

    def _save_split_model(self, save_path):
        """Save split model as external data for layer wise quantization.

        Args:
            save_path (str): the path to save the split model
        """
        if os.path.exists(save_path + "_data"):
            os.remove(save_path + "_data")
        self._model_path = save_path
        onnx.save_model(
            self.model,
            save_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(save_path) + "_data",
            size_threshold=1024,
            convert_attribute=False,
        )

    def _remove_unused_input_output(self):
        """Remove unused input & output for split model."""
        remove_outputs = []
        remove_inputs = []
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()
        for output in self.model.graph.output:
            if output.name not in output_name_to_node.keys():
                remove_outputs.append(output)

        for input in self.model.graph.input:
            if input.name not in input_name_to_nodes.keys():
                remove_inputs.append(input)

        for output in remove_outputs:
            self.model.graph.output.remove(output)
        for input in remove_inputs:
            self.model.graph.input.remove(input)

    def remove_unused_init(self):
        """Remove unused init."""
        remov_inits = []
        if len(self._input_name_to_nodes) == 0:
            self._input_name_to_nodes = self.input_name_to_nodes()
        for init in self.model.graph.initializer:
            if init.name not in self._input_name_to_nodes.keys():
                remov_inits.append(init)
        self.remove_initializers(remov_inits)

    def load_model_initializer_by_tensor(self, data_path=None):
        """Load model initializer by tensor.

        Args:
            data_path (str, optional): the directory of saved initializer. Defaults to None.
        """
        if data_path is None:
            data_path = os.path.dirname(self._model_path)
        for init in self.model.graph.initializer:
            if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
                onnx.external_data_helper.load_external_data_for_tensor(init, data_path)

    def write_external_data_to_new_location(self, external_data_location="external.data", overwrite=False):
        """Write external data of merged quantized model to new location to save memory.

        Args:
            external_data_location (str, optional): external data location of merged quantized model.
                                                    Defaults to "external.data".
            overwrite (bool, optional): if True, remove existed externa data. Defaults to False.
        """
        if overwrite and os.path.exists(os.path.join(os.path.dirname(self._model_path), external_data_location)):
            os.remove(os.path.join(os.path.dirname(self._model_path), external_data_location))
        self.load_model_initializer_by_tensor()
        onnx.external_data_helper.convert_model_to_external_data(self.model, location=external_data_location)
        # TODO : if init is already saved, skip write it
        onnx.external_data_helper.write_external_data_tensors(self.model, filepath=os.path.dirname(self._model_path))

    def merge_split_models(self, to_merge_model):
        """Merge two split model into final model."""
        to_merge_model.write_external_data_to_new_location()
        self.add_nodes([node for node in to_merge_model.nodes()])
        self.add_initializers([init for init in to_merge_model.initializer()])
        self.update()

        # add new output
        for output in to_merge_model.graph().output:
            if output.name not in self.output():
                self.model.graph.output.append(output)

        # remove unused output
        remove_output = []
        for output in self.model.graph.output:
            if output.name in to_merge_model.input():
                remove_output.append(output)
        for output in remove_output:
            self.model.graph.output.remove(output)

        # add new input
        for input in to_merge_model.graph().input:
            if (
                input.name not in self.input()
                and input.name not in self.output()
                and input.name not in self._output_name_to_node.keys()
            ):
                self.model.graph.input.append(input)

    def re_org_output(self, origin_output):
        """Re-org output of merged model for layer-wise quantization."""
        outputs = {}
        tmp_remove = []
        for output in self.model.graph.output:
            outputs[output.name] = output
            tmp_remove.append(output)

        for output in tmp_remove:
            self.model.graph.output.remove(output)

        for out_name in origin_output:
            self.model.graph.output.append(outputs[out_name])
