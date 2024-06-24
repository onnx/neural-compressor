#!/bin/bash
set -eo pipefail
source /neural-compressor/.azure-pipelines/scripts/change_color.sh

log_dir="/neural-compressor/.azure-pipelines/scripts/models"

$BOLD_YELLOW && echo "======= creat log_dir =========" && $RESET
if [ -d "${log_dir}/${model}" ]; then
    $BOLD_GREEN && echo "${log_dir}/${model} already exists, don't need to mkdir." && $RESET
else
    $BOLD_GREEN && echo "no log dir ${log_dir}/${model}, create." && $RESET
    cd ${log_dir}
    mkdir ${model}
fi

$BOLD_YELLOW && echo "====== install requirements ======" && $RESET
/bin/bash /neural-compressor/.azure-pipelines/scripts/install_nc.sh
cd "/neural-compressor/examples/nlp/huggingface_model/text_generation/llama/quantization/weight_only"
pip install -r requirements.txt
