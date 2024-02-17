#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

models=("llama2-7b" "llama2-7b-chat" "llama2-13b" "llama2-13b-chat")
datasets=("stsb" "snli")
output="results_generation_ablation_new"

for dataset in "${datasets[@]}"
do
for model in "${models[@]}"
do
python llm_bias_corpus_ablation.py  \
    --model_name ${model} \
    --dataset ${dataset} \
    --output_folder ${output} \
    --tune_temp \
    # --tune_topp
    # --tune_topk
done
done