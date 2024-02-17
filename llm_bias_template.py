import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
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
    parser.add_argument("--output_folder", type=str, default="results_generation") # model_name = "Llama-2-7b-hf"
    parser.add_argument("--add_instruct", action="store_true", help="add debiasing instruction")
    parser.add_argument("--constrained_decoding", action="store_true", help="add constrained decoding")

    args = parser.parse_args()
    set_seed(42)

    if args.model_name in ["llama2-7b","llama2-7b-chat","llama2-13b","llama2-13b-chat"]:
        model_name = model_dic[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=access_token, device_map="auto",torch_dtype=torch.float16)
    
    elif "vicuna" in args.model_name:
        model_name = model_dic[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",torch_dtype=torch.float16)
    
    elif "falcon" in args.model_name:
        model_name = model_dic[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    elif "mpt" in args.model_name:
        model_name = model_dic[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    elif "opt" in args.model_name:
        model_name = model_dic[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    elif "t5" in args.model_name:
        model_name = model_dic[args.model_name]
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16) 

    
    for key, value in tqdm(dic.items()): # dic = {"occupations": occupations, "personalities": personalities, "hobbies": hobbies, "colors": colors}
        all_logits = []
        results = []
        for val in tqdm(value):
            prompt = add_prompt(key, val)

            if args.add_instruct:
                prompt = f"Continue the sentence without gender mentions: {prompt}"

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            # print(model.generation_config)
            if args.constrained_decoding:
                pronouns = ["she", "her", "hers", "herself", "he", "him", "his", "himself"]
                pronoun_ids = [tokenizer.encode(pro, add_special_tokens=False) for pro in pronouns]
                outputs = model.generate(inputs.input_ids, temperature=0, max_new_tokens=50, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, early_stopping=True, bad_words_ids=pronoun_ids)
            else:
                outputs = model.generate(inputs.input_ids, temperature=0, max_new_tokens=50, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, early_stopping=True)

            generated_sentence = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

            if prompt not in generated_sentence:
                print("prompt not in generated sentence", prompt, "\n", generated_sentence)

            results.append({"input": val, "response": generated_sentence})
            all_logits.append(outputs.scores)
        
        output_dir = f"{args.output_folder}/{args.model_name}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/template_generation_{key}"
        with open(f"{output_file}.json", "w") as f:
            json.dump(results, f)
            f.write('\n')
        torch.save(all_logits, f"{output_file}.pt")



if __name__ == "__main__":
    main()