#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# models=("llama2-7b" "llama2-7b-chat" "llama2-13b" "llama2-13b-chat")
models=("llama2-7b" "llama2-7b-chat")

# models=("llama2-7b")
output="results_generation_ablation_new_chatgpt1"

for model in "${models[@]}"
do
python llm_bias_chatgpt_ablation.py  \
    --model_name ${model} \
    --output_folder ${output} \
    --tune_temp \
    --tune_topp \
    --tune_topk 
done