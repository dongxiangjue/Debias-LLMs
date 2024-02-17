#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2

# models=("llama2-7b" "llama2-7b-chat" "llama2-13b" "llama2-13b-chat" "opt-7b" "opt-13b" "vicuna-7b" "vicuna-13b" "mpt-7b" "mpt-7b-instruct" "mpt-7b-chat" "falcon-7b" "falcon-7b-chat")
models=("llama2-7b" "llama2-7b-chat" "llama2-13b" "llama2-13b-chat")

for model in "${models[@]}"
do
python llm_bias_chatgpt.py \
    --model_name ${model} \
    --output_folder "results_generation_instruct" \
    --add_instruct 
done