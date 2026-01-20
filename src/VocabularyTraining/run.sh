#!/bin/bash

export CUDA_VISIBLE_DEVICES='0'
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

for b in 16
do
            python train_qwen_withreplace.py --dataset_path /path/to/train.txt \
                      --val_dataset_path /path/to/val.txt \
                      --output_dir /path/to/checkpointing/directory/ \
                      --logging_dir /path/to/checkpointing/directory/ \
                      --tokenizer_name_or_path /path/to/Qwen-Vocabulary/folder \
                      --base_tokenizer_path /path/to/base-Qwen/LLM/ \
                      --model_name_or_path /path/to/base-Qwen/LLM/ \
                      --model_type llama3 \
                      --seed 42 \
                      --eval_strategy steps \
                      --logging_steps 5 \
                      --learning_rate 1e-4 \
                      --weight_decay 0.01 \
                      --warmup_ratio 0.05 \
                      --num_train_epochs 3 \
                      --per_device_train_batch_size $b \
                      --per_device_eval_batch_size 8 \
                      --prediction_loss_only \
                      --overwrite_output_dir \
                      --do_train \
                      --do_eval \
                      --lr_scheduler_type cosine \
                      --disable_tqdm False \
                      --label_names labels \
                      --remove_unused_columns False \
                      --save_strategy steps \
                      --save_steps 500 \
                      --bf16 True \
                      --tf32 True \
                      --gradient_checkpointing True \
                      --tune_embeddings \
                      --eval_steps 500 \
                      --ddp_find_unused_parameters True \
                      --gradient_accumulation_steps 4 \
                      --save_total_limit 1 \
                      --load_best_model_at_end True \
                      --metric_for_best_model eval_loss \
                      --greater_is_better False \
                      --save_only_model True \
                      --r 32 \
                      --lora_alpha 64 \
                      --optim adamw_torch_fused


            python train_llama_withreplace.py --dataset_path /path/to/train.txt \
                      --val_dataset_path /path/to/val.txt \
                      --output_dir /path/to/checkpointing/directory/ \
                      --logging_dir /path/to/checkpointing/directory/ \
                      --tokenizer_name_or_path /path/to/LLAMA-vocabulary/folder \
                      --base_tokenizer_path /path/to/LLAMA/LLM/ \
                      --model_name_or_path /path/to/LLAMA/LLM/ \
                      --model_type llama3 \
                      --seed 42 \
                      --eval_strategy steps \
                      --logging_steps 5 \
                      --learning_rate 1e-4 \
                      --weight_decay 0.01 \
                      --warmup_ratio 0.05 \
                      --num_train_epochs 3 \
                      --per_device_train_batch_size $b \
                      --per_device_eval_batch_size 8 \
                      --prediction_loss_only \
                      --overwrite_output_dir \
                      --do_train \
                      --do_eval \
                      --lr_scheduler_type cosine \
                      --disable_tqdm False \
                      --label_names labels \
                      --remove_unused_columns False \
                      --save_strategy steps \
                      --save_steps 500 \
                      --bf16 True \
                      --tf32 True \
                      --gradient_checkpointing True \
                      --tune_embeddings \
                      --eval_steps 500 \
                      --ddp_find_unused_parameters True \
                      --gradient_accumulation_steps 4 \
                      --save_total_limit 1 \
                      --load_best_model_at_end True \
                      --metric_for_best_model eval_loss \
                      --greater_is_better False \
                      --save_only_model True \
                      --r 32 \
                      --lora_alpha 64 \
                      --optim adamw_torch_fused
done