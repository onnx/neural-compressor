#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo "$var" |cut -f2 -d=)
      ;;
      --dataset=*)
          dataset=$(echo "$var" |cut -f2 -d=)
      ;;
      --tokenizer=*)
          tokenizer=$(echo "$var" |cut -f2 -d=)
      ;;
      --algorithm=*)
          algorithm=$(echo "$var" |cut -f2 -d=)
      ;;
      --quant_format=*)
          quant_format=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {

    # Check if the input_model ends with the filename extension ".onnx"
    if [[ $input_model =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        input_model=$(dirname "$input_model")
    fi

    # Check if the output_model ends with the filename extension ".onnx"
    if [[ $output_model =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        output_model=$(dirname "$output_model")
    fi

    # Check if the directory exists
    if [ ! -d "$output_model" ]; then
        # If the directory doesn't exist, create it
	mkdir -p "$output_model"
	echo "Created directory $output_model"
    fi

    extra_cmd=""

    if [[ "${tokenizer}" =~ "Phi-3-mini" ]]; then
        nodes_to_exclude="/model/layers.*/self_attn/qkv_proj/MatMul /model/layers.*/mlp/down_proj/MatMul"
        extra_cmd=$extra_cmd"--nodes_to_exclude ${nodes_to_exclude} --trust_remote_code True "
    fi
    if [[ "${tokenizer}" =~ "Llama-3-8B" ]]; then
        nodes_to_exclude="/model/layers.*/mlp/down_proj/MatMul"
        extra_cmd=$extra_cmd"--nodes_to_exclude ${nodes_to_exclude} "
    fi
    if [[ "${tokenizer}" =~ "Qwen2-7B" ]]; then
        nodes_to_exclude="/model/layers.*/mlp/down_proj/MatMul /model/layers.*/mlp/up_proj/MatMul"
        extra_cmd=$extra_cmd"--nodes_to_exclude ${nodes_to_exclude} "
    fi

    if [ "${tokenizer}" ]; then
	extra_cmd=$extra_cmd"--tokenizer ${tokenizer} "
    fi
    if [ "${batch_size}" ]; then
	extra_cmd=$extra_cmd"--batch_size ${batch_size} "
    fi
    if [ "${dataset}" ]; then
	extra_cmd=$extra_cmd"--dataset ${dataset} "
    fi
    if [ "${algorithm}" ]; then
	extra_cmd=$extra_cmd"--algorithm ${algorithm} "
    fi
    if [ "${tasks}" ]; then
	extra_cmd=$extra_cmd"--tasks ${tasks} "
    fi
    if [ "${quant_format}" ]; then
	extra_cmd=$extra_cmd"--quant_format ${quant_format} "
    fi

    extra_cmd=$extra_cmd"--layer_wise --tune"
    eval "python main.py --model_path ${input_model} --output_model ${output_model} ${extra_cmd}"
}

main "$@"
