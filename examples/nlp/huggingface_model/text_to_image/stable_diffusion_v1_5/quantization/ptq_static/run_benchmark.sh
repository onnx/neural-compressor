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
      --quantized_unet_path=*)
          quantized_unet_path=$(echo "$var" |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo "$var" |cut -f2 -d=)
      ;;
      --prompt=*)
	  prompt=$(echo "$var" |cut -f2 -d=)
      ;;
      --image_path=*)
	  image_path=$(echo "$var" |cut -f2 -d=)
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

    if [ "$quantized_unet_path" ]; then
        extra_cmd=$extra_cmd"--quantized_unet_path=${quantized_unet_path} "
    fi

    if [ "$prompt" ]; then
	extra_cmd=$extra_cmd"--prompt=${prompt} "
    fi

    if [ "$image_path" ]; then
	extra_cmd=$extra_cmd"--image_path=${image_path} "
    fi

    python main.py \
            --model_path="${input_model}" \
            --batch_size="${batch_size-1}" \
            --benchmark \
	    ${extra_cmd}
            
}

main "$@"

