from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import Dataset
import torch
import os
from tqdm import tqdm 
import argparse
import pandas as pd
import pickle
import logging
from transformers.utils import is_flash_attn_2_available
import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"]='expandable_segments:True'
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to the input file for inference"
)

parser.add_argument(
    "--device_id", 
    type=str, 
    default="0",
    required=True, 
    help="Device ID to use for inference (default: 0)"
)

parser.add_argument(
    "--model_path", 
    type=str,
    required=True,
    default="../inference-models/Llama-3.2-3B/", 
    help="Path to the model directory (default: ../inference-models/Llama-3.2-3B/)"
)

parser.add_argument(
    "--eval_column", 
    type=str,
    required=True,
    default="GS_BASE_Llama_1", 
    help="Column name in the evaluation file (default: GS_BASE_Llama_1)"
)

parser.add_argument(
    "--batch_size",
    type=int,
    required=True,
    default=8,
    help="Batch size for inference (default: 8)"
)

parser.add_argument(
    "--prompt_file_prefix",
    type=str,
    required=True,
    default="relevance",
    help="Path prefix to the prompt file (default: relevance)"
)

args = parser.parse_args()

model_path = args.model_path
device_id = args.device_id
prompt_file_prefix = args.prompt_file_prefix
batch_size = args.batch_size
eval_column = args.eval_column
input_file = args.input_file

df = pd.read_csv(input_file)

list_sd = df['SD'].tolist()
list_gs = df[eval_column].tolist()

os.environ["CUDA_VISIBLE_DEVICES"] = device_id
torch.set_float32_matmul_precision('high')

# Load prompts
sys_prompt = open(f'./system_prompt_{prompt_file_prefix}.txt','r', encoding='utf-8').read().strip()
user_prompt = open(f'./user_prompt_{prompt_file_prefix}.txt','r', encoding='utf-8').read().strip()

def format_message(sd,gs):
    """Format message based on model type"""
    return [
            {
                "role": "system", 
                "content": sys_prompt
            },
            {
                "role": "user", 
                "content": user_prompt.replace("[SD]", sd.strip()).replace("[GS]", gs.strip())}
    ]

def extract_response(generated_text):
    """Extract the response from generated text based on model type"""
    return generated_text[-1]['content']

def extract_user_query(generated_text):
    """Extract the user query from generated text based on model type"""
    return generated_text[-2]['content']

# Create messages for all prompts
messages = [format_message(sd, gs) for sd, gs in zip(list_sd, list_gs)]

print(f"Loaded {len(messages)} prompts from {input_file}")
print("Sample prompt:", messages[0])


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation= 'sdpa', #'flash_attention_2' if is_flash_attn_2_available() else 'sdpa',
)

model.eval()

# Create pipeline
pipe = pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    device_map='auto',
)

# Set up tokenizer padding
if pipe.model.generation_config.pad_token_id is None:
    print('Setting up a new pad token id....')
    pipe.model.generation_config.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

pipe.tokenizer.padding_side = "left"

all_new_responses = []
for i in tqdm(range(0, len(messages), batch_size), desc="Processing batches"):

    with torch.no_grad():  
    # Process current batch
        batch_results = pipe(
            messages[i:i+batch_size],
            max_new_tokens=100,
            do_sample=True,
            batch_size=batch_size,
            return_full_text=True
        )
    
    # Write to text file and collect responses
    for idx, result in enumerate(batch_results):
        response = extract_response(result[0]['generated_text'])
        
        all_new_responses.append(response)

    if (i+1)%10 == 0:
        torch.cuda.empty_cache()
        gc.collect()
        torch._dynamo.reset()


df[f'Prompt_Response_{eval_column}_{prompt_file_prefix}'] = all_new_responses
df.to_csv(input_file, index=False)

print(f"Processing complete! Results saved to {input_file}")
print(f"Total responses generated: {len(all_new_responses)}")