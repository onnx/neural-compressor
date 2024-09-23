#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --input_model=*)
          input_model=$(echo "$var" |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo "$var" |cut -f2 -d=)
      ;;
      --tokenizer=*)
          tokenizer=$(echo "$var" |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo "$var" |cut -f2 -d=)
      ;;
      --intra_op_num_threads=*)
          intra_op_num_threads=$(echo "$var" |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {

    # Check if the input_model ends with the filename extension ".onnx"
    if [[ $input_model =~ \.onnx$ ]]; then
        # If the string ends with the filename extension, get the path of the file
        input_model=$(dirname "$input_model")
    fi

    extra_cmd=""

    if [[ "${tokenizer}" =~ "Phi-3-mini" ]]; then
        extra_cmd=$extra_cmd"--trust_remote_code True "
    fi

    if [ "${batch_size}" ]; then
	extra_cmd=$extra_cmd"--batch_size ${batch_size} "
    fi
    if [ "${tokenizer}" ]; then
	extra_cmd=$extra_cmd"--tokenizer ${tokenizer} "
    fi
    if [ "${tasks}" ]; then
	extra_cmd=$extra_cmd"--tasks ${tasks} "
    fi
    if [ "${intra_op_num_threads}" ]; then
	extra_cmd=$extra_cmd"--intra_op_num_threads ${intra_op_num_threads} "
    fi

    extra_cmd=$extra_cmd"--benchmark"
    eval "python main.py --model_path ${input_model} --mode ${mode} ${extra_cmd}"

}

main "$@"
