#!/bin/bash


# stage
stage=dpo

# dataset
dataset=hh_rlhf_en

# use ModelScope
export USE_MODELSCOPE_HUB=1

# ModelScope model
path_to_llama_model="modelscope/Llama-2-7b-ms"
sft_path=sft-checkpoint/checkpoint-600
rm_path=rm-checkpoint/checkpoint-600
output_dir=${stage}-checkpoint

# stats
log_steps=5
save_steps=10


CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage $stage \
    --model_name_or_path $path_to_llama_model \
    --do_train \
    --dataset $dataset \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir $sft_path \
    --output_dir $output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps $log_steps \
    --save_steps $save_steps \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
