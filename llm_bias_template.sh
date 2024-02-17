#!/bin/bash
models=("llama2-7b" "llama2-7b-chat" "llama2-13b" "llama2-13b-chat" "opt-7b" "opt-13b" "vicuna-7b" "vicuna-13b" "mpt-7b" "mpt-7b-instruct" "mpt-7b-chat" "falcon-7b" "falcon-7b-chat")

for model in "${models[@]}"
do
python llm_bias_template.py  --model_name ${model}
done