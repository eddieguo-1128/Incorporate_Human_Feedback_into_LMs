#!/bin/bash

stage=rm

dataset=hh_rlhf_en

# use ModelScope
# export USE_MODELSCOPE_HUB=1

 # ModelScope model
path_to_llama_model="Mistral-7B-Instruct-v0.1"

#"modelscope/Llama-2-7b-ms"

output_dir=${stage}-checkpoint
sft_checkpoint=sft-checkpoint/checkpoint-300
rm_checkpoint=rm-checkpoint/checkpoint-600

log_steps=10
save_steps=100
n_epoch=1
lr_scheduler="cosine"

echo "training $sft_path of stage:$stage at $output_dir"
echo "output at $output_dir"
echo "sft at $sft_path"
echo "rm at $rm_path"

mkdir $output_dir

# use wandb

# training
accelerate config

# CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  
    
accelerate launch src/train_bash.py \
    --stage $stage \
    --model_name_or_path $path_to_llama_model \
    --do_train \
    --dataset $dataset \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir $sft_checkpoint \
    --output_dir $output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps $log_steps \
    --save_steps $save_steps \
    --learning_rate 1e-6 \
    --num_train_epochs $n_epoch \
    --plot_loss \
    --fp16 \
    --report_to wandb
