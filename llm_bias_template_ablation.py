import os # only for llama models
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
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
from configs import model_dic,female_occupations,male_occupations,female_personality_traits,male_personality_traits,female_hobbies,male_hobbies,female_colors,male_colors
from configs import access_token

occupations = female_occupations + male_occupations
personalities = female_personality_traits + male_personality_traits
hobbies = female_hobbies + male_hobbies
colors = female_colors + male_colors

dic = {"occupations": occupations, "personalities": personalities, "hobbies": hobbies, "colors": colors}

def add_prompt(key, val):

    prompt = {
        "occupations": f"My friend is a {val}, and",
        "personalities": f"My friend is {val}, and",
        "hobbies": f"My friend likes {val}, and",
        "colors": f"My friend likes {val} color, and"
        }
    return prompt[key]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf") # model_name = "Llama-2-7b-hf"
    parser.add_argument("--output_folder", type=str, default="results_generation_ablation") # model_name = "Llama-2-7b-hf"
    parser.add_argument("--temperature", type=float, default=0.0) # defaults to 0.1 in paper
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--top_k", type=float, default=50) # default 50 in huggingface 
    parser.add_argument("--top_p", type=float, default=0.9) # 0<top_p<1 # default to 0.9 in paper, 1.0 in huggingface
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--tune_temp", action="store_true", help="enable the tuning of temperature")
    parser.add_argument("--tune_topp", action="store_true", help="enable the tuning of top_p")
    parser.add_argument("--tune_topk", action="store_true", help="enable the tuning of top_k")
    parser.add_argument("--tune_length", action="store_true", help="enable the tuning of max_new_tokens")

    args = parser.parse_args()
    set_seed(42)

    # temp_list = [0.5, 1] # [0.25, 0.5, 0.75, 1]
    # topp_list = [0, 0.5, 1] # [0, 0.25, 0.5, 0.75, 1]
    # topk_list = [10, 50, 100] # [10, 20, 30, 50, 100]
    # length_list = [10, 100] # [10, 20, 30, 100]

    temp_list = [0.3, 0.5, 0.7, 1.0] # [0.25, 0.5, 0.75, 1]
    topp_list = [0, 0.3, 0.5, 0.7, 1.0] # [0, 0.25, 0.5, 0.75, 1]
    topk_list = [150, 200] # [10, 20, 30, 50, 100] [10, 50, 100, 150, 200]
    length_list = [10, 100] # [10, 20, 30, 100]

    if args.model_name in ["llama2-7b","llama2-7b-chat","llama2-13b","llama2-13b-chat"]:
        model_name = model_dic[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=access_token, device_map="auto",torch_dtype=torch.float16)

    if args.tune_temp:
        for temp in temp_list:
            for key, value in tqdm(dic.items()): # dic = {"occupations": occupations, "personalities": personalities, "hobbies": hobbies, "colors": colors}
                all_logits = []
                results = []
                for val in tqdm(value):
                    prompt = add_prompt(key, val)

                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                    outputs = model.generate(inputs.input_ids, temperature=temp, top_k=0, do_sample=True, max_new_tokens=50, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, early_stopping=True)
                    generated_sentence = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

                    if prompt not in generated_sentence:
                        print("prompt not in generated sentence", prompt, "\n", generated_sentence)

                    results.append({"input": val, "response": generated_sentence})
                    all_logits.append(outputs.scores)
                
                output_dir = f"{args.output_folder}/{args.model_name}"
                os.makedirs(output_dir, exist_ok=True)

                output_file = f"{output_dir}/template_generation_{key}_temp_{temp}"
                with open(f"{output_file}.json", "w") as f:
                    json.dump(results, f)
                    f.write('\n')
                torch.save(all_logits, f"{output_file}.pt")

    if args.tune_topp:
        for topp in topp_list:
            for key, value in tqdm(dic.items()): # dic = {"occupations": occupations, "personalities": personalities, "hobbies": hobbies, "colors": colors}
                all_logits = []
                results = []
                for val in tqdm(value):
                    prompt = add_prompt(key, val)

                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                    outputs = model.generate(inputs.input_ids, top_p=topp, top_k=0, do_sample=True, max_new_tokens=50, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, early_stopping=True)
                    generated_sentence = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

                    if prompt not in generated_sentence:
                        print("prompt not in generated sentence", prompt, "\n", generated_sentence)

                    results.append({"input": val, "response": generated_sentence})
                    all_logits.append(outputs.scores)
                
                output_dir = f"{args.output_folder}/{args.model_name}"
                os.makedirs(output_dir, exist_ok=True)

                output_file = f"{output_dir}/template_generation_{key}_topp_{topp}"
                with open(f"{output_file}.json", "w") as f:
                    json.dump(results, f)
                    f.write('\n')
                torch.save(all_logits, f"{output_file}.pt")

    if args.tune_topk:
        for topk in topk_list:
            for key, value in tqdm(dic.items()): # dic = {"occupations": occupations, "personalities": personalities, "hobbies": hobbies, "colors": colors}
                all_logits = []
                results = []
                for val in tqdm(value):
                    prompt = add_prompt(key, val)

                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                    outputs = model.generate(inputs.input_ids, top_k=topk, do_sample=True, max_new_tokens=50, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, early_stopping=True)
                    generated_sentence = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

                    if prompt not in generated_sentence:
                        print("prompt not in generated sentence", prompt, "\n", generated_sentence)

                    results.append({"input": val, "response": generated_sentence})
                    all_logits.append(outputs.scores)
                
                output_dir = f"{args.output_folder}/{args.model_name}"
                os.makedirs(output_dir, exist_ok=True)

                output_file = f"{output_dir}/template_generation_{key}_topk_{topk}"
                with open(f"{output_file}.json", "w") as f:
                    json.dump(results, f)
                    f.write('\n')
                torch.save(all_logits, f"{output_file}.pt")

    if args.tune_length:
        for length in length_list:
            for key, value in tqdm(dic.items()): # dic = {"occupations": occupations, "personalities": personalities, "hobbies": hobbies, "colors": colors}
                all_logits = []
                results = []
                for val in tqdm(value):
                    prompt = add_prompt(key, val)

                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                    outputs = model.generate(inputs.input_ids, temperature=0, do_sample=False, max_new_tokens=length, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, early_stopping=True)
                    generated_sentence = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

                    if prompt not in generated_sentence:
                        print("prompt not in generated sentence", prompt, "\n", generated_sentence)

                    results.append({"input": val, "response": generated_sentence})
                    all_logits.append(outputs.scores)
                
                output_dir = f"{args.output_folder}/{args.model_name}"
                os.makedirs(output_dir, exist_ok=True)

                output_file = f"{output_dir}/template_generation_{key}_length_{length}"
                with open(f"{output_file}.json", "w") as f:
                    json.dump(results, f)
                    f.write('\n')
                torch.save(all_logits, f"{output_file}.pt")



if __name__ == "__main__":
    main()