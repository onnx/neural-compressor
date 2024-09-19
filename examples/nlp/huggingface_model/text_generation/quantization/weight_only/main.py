# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation
import argparse
import json
import logging
import os
import random
import time

import datasets
import evaluation
import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from optimum import onnxruntime as optimum_ort
from torch.nn import functional
from torch.utils import data

from onnx_neural_compressor import data_reader
from onnx_neural_compressor.quantization import QuantFormat, config, matmul_nbits_quantizer, tuning

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.WARN
)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model_path", type=str, help="Folder path of pre-trained onnx model")
parser.add_argument("--benchmark", action="store_true", default=False)
parser.add_argument("--tune", action="store_true", default=False, help="whether quantize the model")
parser.add_argument("--output_model", type=str, default=None, help="path of output dircectory")
parser.add_argument(
    "--batch_size",
    default=1,
    type=int,
)
parser.add_argument(
    "--tokenizer", type=str, help="pretrained model name or path of tokenizer files", default="meta-llama/Llama-2-7b-hf"
)
parser.add_argument("--workspace", type=str, help="workspace to save intermediate files", default="nc_workspace")
parser.add_argument(
    "--algorithm",
    type=str,
    default="WOQ_TUNE",
    choices=["WOQ_TUNE", "RTN", "AWQ", "GPTQ"],
    help="weight only algorithm",
)
parser.add_argument(
    "--pad_max",
    default=196,
    type=int,
)
parser.add_argument(
    "--seqlen",
    default=2048,
    type=int,
)
parser.add_argument(
    "--tasks",
    nargs="+",
    default=["lambada_openai"],
    choices=[
        "winogrande",
        "copa",
        "piqa",
        "rte",
        "hellaswag",
        "openbookqa",
        "lambada_openai",
        "lambada_standard",
        "wikitext",
    ],
    type=str,
    help="tasks list for accuracy validation",
)
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--mode", type=str, help="benchmark mode of performance or accuracy")
parser.add_argument("--intra_op_num_threads", type=int, default=24)
parser.add_argument("--trust_remote_code", type=bool, default=False)
parser.add_argument("--layer_wise", action="store_true", default=False)
parser.add_argument(
    "--quantize_lm_head",
    action="store_true",
    default=False,
    help="language modeling head will not be quantized by default. Doesn't take effect when 'algorithm' is 'WOQ_TUNE'",
)
parser.add_argument(
    "--nodes_to_exclude",
    nargs="+",
    default=[],
    help="nodes that will not be quantized. Doesn't take effect when 'algorithm' is 'WOQ_TUNE'",
)
parser.add_argument("--quant_format", type=str, default="QDQ", choices=["QOperator", "QDQ"])
args = parser.parse_args()

if args.tune and not os.path.exists(args.output_model):
    os.makedirs(args.output_model)

# load model
tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
model_config = transformers.AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)


def tokenize_function(examples):
    example = tokenizer(examples["text"])
    return example


def replace_architectures(json_path):
    # replace 'LLaMATokenizer' to lowercase 'LlamaTokenizer'
    # to avoid bug 'Tokenizer class LLaMATokenizer does not exist or is not currently imported.'
    # refer to https://github.com/huggingface/transformers/issues/22222#issuecomment-1477171703
    with open(json_path, "r") as file:
        data = json.load(file)
        if data["architectures"] == ["LLaMATokenizer"]:
            data["architectures"] = ["LlamaForCausalLM"]

    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)


def eval_func(model):
    model_dir = model
    if isinstance(model, str) and model.endswith(".onnx"):
        model_dir = os.path.dirname(model)

    replace_architectures(os.path.join(model_dir, "config.json"))

    eval_args = evaluation.LMEvalParser(
        model="hf",
        model_args="pretrained=" + model_dir + ",tokenizer=" + args.tokenizer,
        batch_size=args.batch_size,
        tasks=",".join(args.tasks),
        provider="CPUExecutionProvider",
        trust_remote_code=args.trust_remote_code,
    )
    results = evaluation.evaluate(eval_args)

    eval_acc = 0
    for task_name in args.tasks:
        if task_name == "wikitext":
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity,none"]))
            eval_acc += results["results"][task_name]["word_perplexity,none"]
        else:
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc,none"]))
            eval_acc += results["results"][task_name]["acc,none"]

    if len(args.tasks) != 0:
        eval_acc /= len(args.tasks)

    return eval_acc


def benchmark(model):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.intra_op_num_threads

    session = optimum_ort.ORTModelForCausalLM.load_model(  # pylint: disable=E1123
        os.path.join(model, "model.onnx"), session_options=sess_options
    )
    inputs_names = session.get_inputs()
    key_value_input_names = [key.name for key in inputs_names if (".key" in key.name) or (".value" in key.name)]
    use_cache = len(key_value_input_names) > 0

    model = optimum_ort.ORTModelForCausalLM(
        session,  # pylint: disable=E1121
        model_config,
        use_cache=True if use_cache else False,
        use_io_binding=True if use_cache else False,
    )

    max_new_tokens = 32
    prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    total_time = 0.0
    num_iter = 100
    num_warmup = 10
    batch_size = 1
    prompt = [prompt] * batch_size
    total_list = []

    for i in range(num_iter):
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_new_tokens=max_new_tokens)
        gen_ids = output
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        toc = time.time()
        print(gen_text, flush=True)
        if i >= num_warmup:
            total_time += toc - tic

    print("\n", "-" * 10, "Summary:", "-" * 10)
    print(args)
    throughput = (num_iter - num_warmup) / total_time
    print("Throughput: {} samples/s".format(throughput))


class AWQDataloader(data_reader.CalibrationDataReader):

    def __init__(self, model_path, pad_max=196, batch_size=1, sub_folder="train", calibration_sampling_size=8):
        self.encoded_list = []
        self.pad_max = pad_max
        self.batch_size = batch_size
        dataset = datasets.load_dataset(args.dataset, split=sub_folder)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        model = onnx.load(model_path, load_external_data=False)
        inputs_names = [input.name for input in model.graph.input]
        key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
        use_cache = len(key_value_input_names) > 0
        self.batch_size = batch_size

        for idx, (input_ids, attention_mask) in enumerate(dataloader):
            if idx + 1 > calibration_sampling_size:
                break
            ort_input = {}
            ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype("int64")
            ort_input["attention_mask"] = attention_mask[:, :-1].detach().cpu().numpy().astype("int64")
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            ort_input["position_ids"] = position_ids[:, :-1].detach().cpu().numpy().astype("int64")
            if use_cache:
                # Create dummy past_key_values for decoder
                num_attention_heads = model_config.num_key_value_heads
                embed_size_per_head = model_config.hidden_size // model_config.num_attention_heads
                shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                key_or_value = np.zeros(shape, dtype=np.float32)
                for key_value_input_name in key_value_input_names:
                    ort_input[key_value_input_name] = key_or_value
            self.encoded_list.append(ort_input)

        self.iter_next = iter(self.encoded_list)

    def collate_batch(self, batch):

        input_ids_padded = []
        attention_mask_padded = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            attention_mask = torch.ones(len(input_ids))
            input_ids = functional.pad(input_ids, (0, pad_len), value=1)
            attention_mask = functional.pad(attention_mask, (0, pad_len), value=0)
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
        return torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.encoded_list)


class GPTQDataloader(data_reader.CalibrationDataReader):

    def __init__(self, model_path, batch_size=1, seqlen=2048, sub_folder="train", calibration_sampling_size=8):
        # large `calibration_sampling_size` may result in long GPTQ running time
        # recommend to use smaller `calibration_sampling_size` value
        random.seed(0)
        self.encoded_list = []

        self.batch_size = batch_size
        traindata = datasets.load_dataset(args.dataset, split=sub_folder)
        traindata = traindata.map(tokenize_function, batched=True)
        traindata.set_format(type="torch", columns=["input_ids", "attention_mask"])

        session = ort.InferenceSession(model_path)
        inputs_names = [input.name for input in session.get_inputs()]
        key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
        use_cache = len(key_value_input_names) > 0

        for i in range(calibration_sampling_size):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = traindata[i]
                if trainenc["input_ids"].shape[0] > seqlen:
                    break
            i = random.randint(0, trainenc["input_ids"].shape[0] - seqlen - 1)
            j = i + seqlen
            inp = trainenc["input_ids"][i:j].unsqueeze(0)
            mask = torch.ones(inp.shape)

            ort_input = {}
            ort_input["input_ids"] = inp.detach().cpu().numpy().astype("int64")
            ort_input["attention_mask"] = mask.detach().cpu().numpy().astype("int64")
            input_shape = ort_input["input_ids"].shape
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
            ort_input["position_ids"] = position_ids.numpy()
            if use_cache:
                # create dummy past_key_values for decoder first generation step
                num_attention_heads = model_config.num_key_value_heads
                embed_size_per_head = model_config.hidden_size // model_config.num_attention_heads
                shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                key_or_value = np.zeros(shape, dtype=np.float32)
                for key_value_input_name in key_value_input_names:
                    ort_input[key_value_input_name] = key_or_value

            self.encoded_list.append(ort_input)
        self.iter_next = iter(self.encoded_list)

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.encoded_list)


if __name__ == "__main__":
    if args.benchmark:
        if args.mode == "performance":
            benchmark(args.model_path)
        elif args.mode == "accuracy":
            acc_result = eval_func(args.model_path)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)

    if args.tune:
        model_name = "model.onnx"  # require optimum >= 1.14.0
        model_path = os.path.join(args.model_path, model_name)
        best_model = None

        nodes_to_exclude = ["/lm_head/MatMul"] if not args.quantize_lm_head else []
        nodes_to_exclude = list(set(args.nodes_to_exclude + nodes_to_exclude))
        quant_format = QuantFormat.QOperator if args.quant_format == "QOperator" else QuantFormat.QDQ
        if args.algorithm.upper() == "RTN":
            algo_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                layer_wise_quant=args.layer_wise, quant_format=quant_format
            )
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model_path,
                n_bits=4,
                block_size=32,
                is_symmetric=True,
                algo_config=algo_config,
                nodes_to_exclude=nodes_to_exclude,
            )
            quant.process()
            best_model = quant.model

        elif args.algorithm.upper() == "AWQ":
            calibration_data_reader = AWQDataloader(model_path, pad_max=args.pad_max, batch_size=1)
            algo_config = matmul_nbits_quantizer.AWQWeightOnlyQuantConfig(
                calibration_data_reader=calibration_data_reader,
                enable_mse_search=False,
                quant_format=quant_format,
            )
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model_path,
                n_bits=4,
                block_size=32,
                is_symmetric=True,
                algo_config=algo_config,
                nodes_to_exclude=nodes_to_exclude,
            )
            quant.process()
            best_model = quant.model

        elif args.algorithm.upper() == "GPTQ":
            calibration_data_reader = GPTQDataloader(model_path, seqlen=args.seqlen, batch_size=1)
            algo_config = matmul_nbits_quantizer.GPTQWeightOnlyQuantConfig(
                calibration_data_reader=calibration_data_reader,
                layer_wise_quant=args.layer_wise,
                quant_format=quant_format,
            )
            quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
                model_path,
                n_bits=4,
                block_size=32,
                is_symmetric=False,
                algo_config=algo_config,
                nodes_to_exclude=nodes_to_exclude,
            )
            quant.process()
            best_model = quant.model

        elif args.algorithm.upper() == "WOQ_TUNE":
            calibration_data_reader = GPTQDataloader(model_path, seqlen=args.seqlen, batch_size=1)
            # set tolerable_loss to 0.5% for test, default is 1%
            custom_tune_config = tuning.TuningConfig(
                config_set=config.get_woq_tuning_config(quant_format=quant_format), tolerable_loss=0.005
            )
            best_model = tuning.autotune(
                model_input=model_path,
                tune_config=custom_tune_config,
                eval_fn=eval_func,
                calibration_data_reader=calibration_data_reader,
            )

        if best_model is not None:
            onnx.save_model(
                best_model,
                os.path.join(args.output_model, model_name),
                save_as_external_data=True,
            )
            model_config.to_json_file(os.path.join(args.output_model, "config.json"), use_diff=False)
