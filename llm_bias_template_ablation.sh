#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

models=("llama2-7b" "llama2-7b-chat" "llama2-13b" "llama2-13b-chat")
output="results_generation_ablation_new"

for model in "${models[@]}"
do
python llm_bias_template_ablation.py  \
    --model_name ${model} \
    --output_folder ${output} \
    --tune_temp
    # --tune_topp \
    # --tune_topk 
done