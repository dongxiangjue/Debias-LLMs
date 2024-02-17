import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
import torch
from transformers import pipeline
import argparse
import re
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import json
import numpy as np
from transformers import set_seed
from configs import model_dic
from peft import AutoPeftModelForCausalLM
from configs import access_token

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="stsb")
    parser.add_argument("--model_name", type=str, default="llama2-7b") # model_name = "Llama-2-7b-hf"
    parser.add_argument("--model_dir", type=str, default="results_lora_2e-05_1.0_1_fullloss") # model_name = "Llama-2-7b-hf"
    parser.add_argument("--output_folder", type=str, default="results_generation") # model_name = "Llama-2-7b-hf"

    args = parser.parse_args()
    model_dir = f"{args.model_dir}/{args.model_name}/final_checkpoint"

    model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map="auto",torch_dtype=torch.bfloat16, load_in_4bit=True, use_auth_token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    in_path = f"{args.dataset}_neutral_test.csv"

    df = pd.read_csv(in_path, names=["text"])
    items = df["text"].tolist()
    print(len(items))

    print(items)

    all_logits = []
    results = []
    for item in tqdm(items):

        prompt = f"My friend {item}, and"  # modified

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        outputs = model.generate(input_ids=inputs.input_ids, temperature=0, max_new_tokens=50, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, early_stopping=True)

        generated_sentence = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # print(generated_sentence)
        # output_sentence = generated_sentence.split("### Response:")[1].strip()
        if prompt not in generated_sentence:
            print("prompt not in generated sentence", prompt, "\n", generated_sentence)
        # print(tokenizer.batch_decode(torch.topk(outputs.scores[0], k=5)[1]))
        # print(torch.topk(outputs.scores[0], k=5))
        # print(output_sentence)

        results.append({"input": item, "response": generated_sentence})
        all_logits.append(outputs.scores)
    
    output_dir = f"{args.output_folder}_{args.model_dir}/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/corpus_generation_{args.dataset}"
    with open(f"{output_file}.json", "w") as f:
        json.dump(results, f)
        f.write('\n')
    torch.save(all_logits, f"{output_file}.pt")

if __name__ == "__main__":
    main()