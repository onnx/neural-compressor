#!/bin/bash
set -eo pipefail
set -xe
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"; do
    case $i in
    --stage=*)
        stage=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --model=*)
        model=$(echo $i | sed "s/${PATTERN}//")
        ;;
    *)
        echo "Parameter $i not recognized."
        exit 1
        ;;
    esac
done

CONFIG_PATH=/neural-compressor/examples/.config/model_params_onnxrt.json
model_src_dir=/neural-compressor/examples/nlp/huggingface_model/text_generation/llama/quantization/weight_only
dataset_location=NeelNanda/pile-10k
input_model=/tf_dataset2/models/huggingface/opt-125m
batch_size=16

function run_prepare_model() {
    python prepare_model.py --input_model="$model" --output_model="./model_export" --task=text-generation-with-past
}

function run_quantize() {
    bash run_quant.sh --input_model="./model_export" \
        --output_model="./model_tune" \
        --batch_size="$batch_size" \
        --dataset=NeelNanda/pile-10k \
        --tokenizer="$model" \
        --algorithm=WOQ_TUNE
}

function run_accuray() {
    bash run_benchmark.sh --input_model="./model_tune" \
        --batch_size="$batch_size" \
        --mode=accuracy \
        --tokenizer="$model" \
        --tasks=lambada_openai | tee -a accuracy.log
}

function main() {
    cd $model_src_dir
    if [ "$stage" == "prepare_model" ]; then
        run_prepare_model
    elif [ "$stage" == "quantize" ]; then
        run_quantize
    elif [ "$stage" == "accuracy" ]; then
        run_accuray
    else
        exit 1
    fi
}

main
