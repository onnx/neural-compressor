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
model_src_dir=$(jq '."onnxrt"."$model"."model_src_dir"' $CONFIG_PATH)
dataset_location=$(jq '."onnxrt"."$model"."dataset_location"' $CONFIG_PATH)
input_model=$(jq '."onnxrt"."$model"."input_model"' $CONFIG_PATH)

function run_prepare_model() {
    if [ -f "$input_model" ]; then
        echo "model exists"
    else
        echo "model not found" && exit 1
    fi
}

function run_quantize() {
    bash run_quant.sh --input_model="$input_model" \
        --dataset_location=$dataset_location \
        --label_path="$model" \
        --output_model="./model_tune"
}

function run_accuracy() {
    bash run_benchmark.sh --input_model="./model_tune" \
        --dataset_location=$dataset_location \
        --label_path="$model" \
        --mode=$1 | tee -a accuracy.log
}

function main() {
    cd $model_src_dir
    if [ "$stage" == "prepare_model" ]; then
        run_prepare_model
    elif [ "$stage" == "quantize" ]; then
        run_quantize
    elif [ "$stage" == "accuracy" ]; then
        run_accuracy "accuracy"
    elif [ "$stage" == "performance" ]; then
        run_accuracy "performance"
    else
        exit 1
    fi
}

main
