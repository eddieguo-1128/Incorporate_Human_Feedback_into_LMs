#!/bin/bash


stage=ppo

dataset=hh_rlhf_en

sft_path=hh-rlhf-sft
rm_path=hh-rlhf-rm-open-llama-3b
output_dir=${sft_path}-${stage}-checkpoint

log_steps=10
save_steps=1000
lr=5e-5
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
    --output_dir $output_dir \
    --model_name_or_path $sft_path \
    --do_train \
    --dataset $dataset \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --lr_scheduler_type $lr_scheduler \
    --top_k 0 \
    --top_p 0.9 \
    --logging_steps $log_steps \
    --save_steps $save_steps \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16 \
    --reward_model $rm_path

echo "experiment finish"
