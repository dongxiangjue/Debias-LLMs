# Disclosure and Mitigation of Gender Bias in LLMs
Anonymous code repository for paper submission.

## Dataset
LLM-generated inputs: `templates_200.txt` and `templates_200_1.txt`

Naturally-sourced inputs: adapted from STS-B and SNLI datasets, `stsb_neutral_test.csv` and `snli_neutral_test.csv`, Apache-2.0 license.

## Models
|Model|Link|License|
|---|---|---|
|LLaMA2 7B|https://huggingface.co/meta-llama/Llama-2-7b-hf|LLAMA 2 Community License|
|LLaMA2 7B-Chat|https://huggingface.co/meta-llama/Llama-2-7b-chat-hf|LLAMA 2 Community License|
|LLaMA2 13B|https://huggingface.co/meta-llama/Llama-2-13b-hf|LLAMA 2 Community License|
|LLaMA2 13B-Chat|https://huggingface.co/meta-llama/Llama-2-13b-chat-hf|LLAMA 2 Community License|
|Vicuna 7B|https://huggingface.co/lmsys/vicuna-7b-v1.5|LLAMA 2 Community License|
|Vicuna 13B|https://huggingface.co/lmsys/vicuna-13b-v1.5|LLAMA 2 Community License|
|Falcon 7B|https://huggingface.co/tiiuae/falcon-7b|Apache 2.0 License|
|Falcon 7B-Instruct|https://huggingface.co/tiiuae/falcon-7b-instruct|Apache 2.0 License|
|OPT 6.7B|https://huggingface.co/facebook/opt-6.7b|OPT-175B License|
|OPT 13B|https://huggingface.co/facebook/opt-13b|OPT-175B License|

## Probing
`llm_bias_template.sh`
`llm_bias_chatgpt.sh`
`llm_bias_corpus.sh`
## Hyperparameter Tuning
`llm_bias_template_ablation.sh`
`llm_bias_chatgpt_ablation.sh`
`llm_bias_corpus_ablation.sh`
## Instruction Guiding
`llm_bias_template_instruct.sh`
`llm_bias_chatgpt_instruct.sh`
`llm_bias_corpus_instruct.sh`
## Debias Tuning
`llm_bias_template_lora.sh`
`llm_bias_chatgpt_lora.sh`
`llm_bias_corpus_lora.sh`
