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
      --alpha=*)
          alpha=$(echo "$var" |cut -f2 -d=)
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

    # Check if the directory exists
    if [ ! -d $(dirname "$output_model") ]; then
        # If the directory doesn't exist, create it
	mkdir -p $(dirname "$output_model")
	echo "Created directory $(dirname $output_model)"
    fi

    python main.py \
            --model_path "${input_model}" \
            --output_model "${output_model}" \
	    --alpha "${alpha-0.7}" \
            --tune
}

main "$@"

