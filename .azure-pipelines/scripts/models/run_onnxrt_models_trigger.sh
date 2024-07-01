#!/bin/bash
set -eo pipefail
set -xe
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"; do
    case $i in
    --stage=*)
        stage=${i//${PATTERN}/}
        ;;
    --model=*)
        model=${i//${PATTERN}/}
        ;;
    *)
        echo "Parameter $i not recognized."
        exit 1
        ;;
    esac
done

CONFIG_PATH="/neural-compressor/examples/.config/model_params_onnxrt.json"
model_src_dir=$(jq -r ".\"onnxrt\".\"$model\".\"model_src_dir\"" "$CONFIG_PATH")
if [[ "$model" == *"resnet"* ]]; then
    dataset_location="/tf_dataset2/datasets/imagenet/ImagenetRaw/ImagenetRaw_small_5000/ILSVRC2012_img_val"
    label_path="/tf_dataset2/datasets/imagenet/ImagenetRaw/ImagenetRaw_small_5000/val.txt"
else
    dataset_location=$(jq -r ".\"onnxrt\".\"$model\".\"dataset_location\"" "$CONFIG_PATH")
fi

input_model=$(jq -r ".\"onnxrt\".\"$model\".\"input_model\"" "$CONFIG_PATH")

function run_prepare_model() {
    if [ -f "$input_model" ]; then
        echo "model exists"
    else
        echo "model not found" && exit 1
    fi
}

function run_quantize() {
    if [ "$model" == "bert-base-uncased" ]; then
        bash run_quant.sh --input_model="$input_model" \
            --dataset_location="$dataset_location" \
            --label_path="$model" \
            --output_model="./model_tune" \
            --quant_format="QDQ"
    else
        bash run_quant.sh --input_model="$input_model" \
            --dataset_location="$dataset_location" \
            --label_path="$model" \
            --output_model="./model_tune"
    fi
}

function run_accuracy() {
    bash run_benchmark.sh --input_model="./model_tune" \
        --dataset_location="$dataset_location" \
        --label_path="$model" \
        --mode="accuracy" \
        --batch_size="16" | tee -a accuracy.log
}

function run_performance() {
    bash run_benchmark.sh --input_model="./model_tune" \
        --dataset_location="$dataset_location" \
        --label_path="$model" \
        --mode="performance" \
        --intra_op_num_threads="8" \
        --batch_size="1" | tee -a accuracy.log
}

function main() {
    cd "/neural-compressor/examples/$model_src_dir"
    if [ "$stage" == "prepare_model" ]; then
        run_prepare_model
    elif [ "$stage" == "quantize" ]; then
        run_quantize
    elif [ "$stage" == "accuracy" ]; then
        run_accuracy
    elif [ "$stage" == "performance" ]; then
        run_performance
    else
        echo "invalid stage: $stage" && exit 1
    fi
}

main
