Step-by-Step
============

This example confirms llama's weight only accuracy on [lambada](https://huggingface.co/datasets/lambada).

# Prerequisite

## 1. Environment
```shell
pip install onnx-neural-compressor
pip install -r requirements.txt
```
> Note: Validated ONNX Runtime [Version](/docs/installation_guide.md#validated-software-environment).

## 2. Prepare Model

Note that this README.md uses meta-llama/Llama-2-7b-hf as an example. We verified weight-only quantization on other models as follows.

| Model | Num Hidden Layers| Num Attention Heads | Hidden Size |
| --- | --- | --- | --- |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 32 | 32 | 4096 |
| [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | 32 | 32 | 4096 |
| [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 40 | 40 | 5120 |
| [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | 40 | 40 | 5120 |
| [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf) | 80 | 64 | 8192 |
| [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | 80 | 64 | 8192 |
| [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | 32 | 32 | 4096 |
| [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) | 32 | 32 | 3072 |
| [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | 28 | 28 | 3584 |

Export to ONNX model:
```bash
python prepare_model.py  --input_model="meta-llama/Llama-2-7b-hf" \
                         --task=text-generation-with-past \ # or text-generation
```


# Run

## 1. Quantization

Set `algorithm=WOQ_TUNE` to tune weight-only quantization algorithm or specify algorithm to `RTN` or `GPTQ` or `AWQ`.

`quant_format=QDQ` works only when:
- onnxruntime >= 1.19.0
- opset version of the model >= 21
- quantized bits is in [4, 8]

otherwise it will execute QOperator automatically.

```bash
bash run_quant.sh --input_model=/path/to/model \ # folder path of onnx model
                  --output_model=/path/to/model_tune \ # folder path to save onnx model
                  --batch_size=batch_size # optional \
                  --dataset=NeelNanda/pile-10k \
                  --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                  --algorithm=WOQ_TUNE # support WOQ_TUNE, RTN, AWQ, GPTQ \
                  --quant_format=QOperator # support QOperator and QDQ
```

## 2. Benchmark

Accuracy:

```bash
bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                      --batch_size=batch_size \ # optional
                      --mode=accuracy \
                      --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                      --tasks=lambada_openai
```

Performance:
```bash
numactl -m 0 -C 0-23 bash run_benchmark.sh --input_model=path/to/model \ # folder path of onnx model
                                           --mode=performance \
                                           --batch_size=batch_size # optional \
                                           --tokenizer=meta-llama/Llama-2-7b-hf \ # model name or folder path containing all relevant files for model's tokenizer
                                           --intra_op_num_threads=24

```
