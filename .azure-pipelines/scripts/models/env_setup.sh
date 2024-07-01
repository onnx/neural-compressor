#!/bin/bash
set -eo pipefail
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"; do
    case $i in
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

# log_dir="/neural-compressor/.azure-pipelines/scripts/models"

# $BOLD_YELLOW && echo "======= creat log_dir =========" && $RESET
# if [ -d "${log_dir}/${model}" ]; then
#     $BOLD_GREEN && echo "${log_dir}/${model} already exists, don't need to mkdir." && $RESET
# else
#     $BOLD_GREEN && echo "no log dir ${log_dir}/${model}, create." && $RESET
#     cd ${log_dir}
#     mkdir ${model}
# fi

$BOLD_YELLOW && echo "====== install ONC ======" && $RESET
cd /neural-compressor
source .azure-pipelines/scripts/change_color.sh
/bin/bash .azure-pipelines/scripts/install_nc.sh

$BOLD_YELLOW && echo "====== install requirements ======" && $RESET
cd "/neural-compressor/examples/$model_src_dir"
pip install -r requirements.txt
pip list
