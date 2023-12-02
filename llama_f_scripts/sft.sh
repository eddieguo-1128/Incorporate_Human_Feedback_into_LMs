#!/bin/bash


stage=sft

dataset=hh_rlhf_en

path_to_llama_model = 

output_dir=${stage}-checkpoint

log_steps=10
save_steps=1000
n_epoch=3.0
batch_size=4
gradient_accumulation_steps=4
lr_scheduler=cosine

echo "training $sft_path of stage:$stage at $output_dir"
echo "output at $output_dir"
echo "sft at $sft_path"
echo "rm at $rm_path"

mkdir $output_dir

    CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage $stage \
    --model_name_or_path $path_to_llama_model \
    --do_train \
    --dataset $dataset \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $output_dir \
    --overwrite_cache \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --lr_scheduler_type $scheduler \
    --logging_steps $log_steps \
    --save_steps $save_steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16

echo "experiment finish"
