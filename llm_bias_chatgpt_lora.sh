#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

models=("llama2-7b" "llama2-7b-chat" "llama2-13b" "llama2-13b-chat")
# models=("llama2-7b-chat" "llama2-13b" "llama2-13b-chat")

model_dir="results_lora_0.0002_1.0_500_0201_total_loss_16"

for model in "${models[@]}"
do
python llm_bias_chatgpt_lora.py --model_name ${model} --model_dir ${model_dir}
done