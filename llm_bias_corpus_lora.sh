#!/bin/bash
# export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=1

datasets=("stsb" "snli")
# datasets=("stsb")
# datasets=("snli")

models=("llama2-7b" "llama2-7b-chat" "llama2-13b" "llama2-13b-chat")
# models=("llama2-7b-chat" "llama2-13b" "llama2-13b-chat")

model_dir="results_lora_0.0002_1.0_500_0201_nokl_loss_16"

for dataset in "${datasets[@]}"
do
for model in "${models[@]}"
do
python llm_bias_corpus_lora.py --model_name ${model} --dataset ${dataset} --model_dir ${model_dir}
done
done