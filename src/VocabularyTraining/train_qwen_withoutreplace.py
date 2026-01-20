import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datasets
import torch
import torch.nn as nn
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer)

from transformers import EarlyStoppingCallback

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config import CustomArgumentParser

import json
import math
import numpy as np

hf_token = "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"

def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size

def instantiate_model_by_mean(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    tie_word_embeddings: bool = False
):
    # Determine the device (GPU or CPU)
    print("Inside instantiate model by mean")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    
    # Initialize source embeddings
    source_embeddings = source_model.get_input_embeddings().weight#.detach().to(device)
    target_embeddings = torch.zeros(
        (round_to_nearest_multiple(len(target_tokenizer), 8), 
         source_embeddings.shape[1]), device=device
    )
    
    source_end = source_vocab['<|endoftext|>']
    target_end = target_vocab['<|endoftext|>']
    
    #PLM-Part
    target_embeddings[:source_end] = source_embeddings[:source_end]
    #Reserved Tokens
    for idx in range(target_end,len(target_tokenizer)):
        target_embeddings[idx] = source_embeddings[source_tokenizer.convert_tokens_to_ids(target_tokenizer.convert_ids_to_tokens(idx))]

    if not tie_word_embeddings:
        print("You are using the output projection init.")
        source_head_embeddings = source_model.get_output_embeddings().weight#.detach().to(device)
        target_head_embeddings = torch.zeros(
            (round_to_nearest_multiple(len(target_tokenizer), 8), 
             source_head_embeddings.shape[1]), device=device
        )
        
        #PLM-Part
        target_head_embeddings[:source_end] = source_head_embeddings[:source_end]
        #Reserved Tokens        
        for idx in range(target_end,len(target_tokenizer)):
            target_head_embeddings[idx] = source_head_embeddings[source_tokenizer.convert_tokens_to_ids(target_tokenizer.convert_ids_to_tokens(idx))]

    # Initialize the rest of the embeddings
    for i in range(source_end, target_end):
        token = target_tokenizer.convert_ids_to_tokens(i)
        if token.startswith('Ġ'): token = ' '+token[1:]
        source_ids = source_tokenizer.convert_tokens_to_ids(source_tokenizer.tokenize(token))
        source_ids = torch.tensor(source_ids)#, device=device)
        target_embeddings[i] = source_embeddings[source_ids].mean(dim=0)
        # print(i,token, target_embeddings[i], source_ids)
        if not tie_word_embeddings:
            target_head_embeddings[i] = source_head_embeddings[source_ids].mean(dim=0)

    # Expand the embeddings
    target_model = source_model #AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    target_model.resize_token_embeddings(
        len(target_tokenizer), 
        pad_to_multiple_of=8  # See https://github.com/huggingface/transformers/issues/26303
    )
    target_model.get_input_embeddings().weight.data = target_embeddings
    target_model.config.vocab_size = round_to_nearest_multiple(len(target_tokenizer), 8)
    
    if not tie_word_embeddings:
        target_model.get_output_embeddings().weight.data = target_head_embeddings

    return target_model, target_tokenizer


import re
from typing import List, Dict, Any


import re
from typing import List, Dict, Any

def simple_logical_chunking(
    examples: Dict[str, List[str]],
    tokenizer,
    max_tokens: int = 512,
    pad_id: int | None = None,
) -> Dict[str, List[List[int]]]:
    """
    Chunk text based on sentence boundaries.
    
    For each text:
      - If entire text fits in max_tokens, pad and return
      - Otherwise, split into sentences and build chunks
      - Each chunk is padded to max_tokens
      - Long sentences (>max_tokens) are split into blocks
      - All text is preserved
    
    Returns:
      {"input_ids": List[List[int]]}  # each length == max_tokens
    """
    if pad_id is None:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    # Check which special tokens the tokenizer has
    has_bos = tokenizer.bos_token_id is not None
    has_eos = tokenizer.eos_token_id is not None
    special_tokens_count = (1 if has_bos else 0) + (1 if has_eos else 0)

    def add_special_tokens_manual(token_ids):
        """Add special tokens based on what tokenizer supports"""
        result = token_ids.copy()
        if has_bos:
            result = [tokenizer.bos_token_id] + result
        if has_eos:
            result = result + [tokenizer.eos_token_id]
        return result

    final_chunks: List[List[int]] = []

    for text in examples["text"]:
        text = (text or "").strip()
        if not text:
            continue
        
        # Try to tokenize entire text first
        text_ids = tokenizer.encode(text, add_special_tokens=True)
        
        # If it fits, just pad and continue
        if len(text_ids) <= max_tokens:
            pad_len = max_tokens - len(text_ids)
            final_chunks.append(text_ids + [pad_id] * pad_len)
            continue
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk_text = ""
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Build candidate chunk text (with space if needed)
            if current_chunk_text:
                candidate_text = current_chunk_text + " " + sent
            else:
                candidate_text = sent
            
            # Tokenize to check length
            candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=True)
            
            # If this sentence alone is too long, handle it specially
            sent_ids = tokenizer.encode(sent, add_special_tokens=True)
            if len(sent_ids) > max_tokens:
                # Finalize current chunk if any
                if current_chunk_text:
                    chunk_ids = tokenizer.encode(current_chunk_text, add_special_tokens=True)
                    pad_len = max_tokens - len(chunk_ids)
                    final_chunks.append(chunk_ids + [pad_id] * pad_len)
                    current_chunk_text = ""
                
                # Split long sentence into blocks
                # Remove special tokens for splitting
                sent_ids_no_special = tokenizer.encode(sent, add_special_tokens=False)
                effective_max = max_tokens - special_tokens_count
                
                for i in range(0, len(sent_ids_no_special), effective_max):
                    block = sent_ids_no_special[i:i + effective_max]
                    block_ids = add_special_tokens_manual(block)
                    pad_len = max_tokens - len(block_ids)
                    final_chunks.append(block_ids + [pad_id] * pad_len)
                
                continue
            
            # If adding this sentence exceeds limit, finalize current chunk
            if len(candidate_ids) > max_tokens:
                chunk_ids = tokenizer.encode(current_chunk_text, add_special_tokens=True)
                pad_len = max_tokens - len(chunk_ids)
                final_chunks.append(chunk_ids + [pad_id] * pad_len)
                current_chunk_text = sent  # Start new chunk with this sentence
            else:
                # Add sentence to current chunk
                current_chunk_text = candidate_text
        
        # Finalize any remaining chunk
        if current_chunk_text:
            chunk_ids = tokenizer.encode(current_chunk_text, add_special_tokens=True)
            pad_len = max_tokens - len(chunk_ids)
            final_chunks.append(chunk_ids + [pad_id] * pad_len)

    return {"input_ids": final_chunks}

def main(args, training_args):
    model_id = args.model_name_or_path
    logger.info(f"Device: {torch.cuda.device_count()}")
    # Load the dataset
    train_dataset = datasets.load_dataset("text",data_files={"train":args.dataset_path},split='train',cache_dir="./")

    #train_dataset = datasets.load_from_disk(args.dataset_path)
    #print("train dataset: ",train_dataset)
    #exit()

    if args.val_dataset_path is not None:
        val_dataset = datasets.load_dataset("text",data_files={"validation":args.val_dataset_path},split='validation',cache_dir="./")
    else:
        val_dataset = None

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    source_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer_path)
    
    # Set up the data collator
    tokenizer.pad_token = tokenizer.eos_token
    
    #Tokenize Train set
    train_tokenized_dataset = train_dataset.map(
        lambda examples: simple_logical_chunking(examples, tokenizer, 512, tokenizer.eos_token_id),
        batched=True, 
        num_proc=16,
        remove_columns=train_dataset.column_names,
    )

    #Tokenize Train set
    val_tokenized_dataset = val_dataset.map(
        lambda examples: simple_logical_chunking(examples, tokenizer, 512, tokenizer.eos_token_id),
        batched=True, 
        num_proc=16,
        remove_columns=train_dataset.column_names,
    )

    #Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load the model
    if args.freeze_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #device_map='sequential',#'auto',
            #max_memory=max_memory,
            token=hf_token
        )        
        model.resize_token_embeddings(len(tokenizer))

    else:
        print("Inside weight training...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,            
            token=hf_token
        )
        
        model,tokenizer = instantiate_model_by_mean(model, source_tokenizer, tokenizer, False)
        # model = prepare_model_for_kbit_training(model)
        
#     for param in model.parameters():
# #        param.requires_grad = False  # freeze the model - train adapters later
#         if param.ndim == 1:
#             # cast the small parameters (e.g. layernorm) to fp32 for stability
#             param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    #model.lm_head = CastOutputToFloat(model.lm_head)
    logger.info(model)

    # Set up LoRA
    if not args.no_lora:
        logger.info(f'Before PEFT applied (Memory): {model.get_memory_footprint()}')
        
        if args.model_type in ("llama3", "mistral"):
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        else:
            raise ValueError(f"Model type {args.model_type} not supported.")
        
        if args.tune_embeddings:
            if args.model_type in ("llama3", "mistral"):
                modules_to_save = ["lm_head", "embed_tokens"]
            else:
                raise ValueError(f"Model type {args.model_type} not supported.")
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                target_modules=target_modules,
                inference_mode=False, 
                r=args.r,
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)

        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=args.r,
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
            )
            model = get_peft_model(model, peft_config)

        logger.info(f'After PEFT applied (Memory): {model.get_memory_footprint()}')

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for par_name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                print("Trainable: ",par_name,'||',param.numel())
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        
    print_trainable_parameters(model)
    
    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        data_collator=data_collator,
        eval_dataset=val_tokenized_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            
        ],
    )

    # Train the model
    trainer.train()

    # Save the model
    # trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = CustomArgumentParser()
    args, training_args = parser.parse_args()
    logger.info(args)
    logger.info(training_args)
    main(args, training_args)