#!/bin/bash

model_name=${1}
script_path="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

if [ -z "$model_name" ]; then
    echo "Usage: ${0} <model_name>"
    exit 1
fi

app_file="${script_path}/example-app"
model_dir="${script_path}/model"
model_file="${model_name}.onnx"
model_path="${model_dir}/${model_file}"
link_file="${model_dir}/model.onnx"

if [ ! -e "${model_path}" ]; then
    echo "Error: Model file '${model_path}' not found."
    exit 1
fi

if [ ! -e "${app_file}" ]; then
    echo "Error: App executable '${app_file}' not found."
    exit 1
fi

if [ -e "${link_file}" ]; then
    rm -f "${link_file}"
fi
ln -sf "${model_file}" "${link_file}"

echo "Running app..."
"${app_file}"
