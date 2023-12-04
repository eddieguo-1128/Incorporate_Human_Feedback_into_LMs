#!/bin/bash


stage=ppo

dataset=hh_rlhf_en

# ModelScope model
path_to_llama_model="modelscope/Llama-2-7b-ms"
sft_path=sft-checkpoint/checpoint-600
rm_path=hh-rlhf-rm-open-llama-3b
output_dir=${stage}-checkpoint

log_steps=10
save_steps=100
lr=5e-5
n_epoch=1
batch_size=4
gradient_accumulation_steps=4
lr_scheduler=cosine

echo "training stage:$stage at $output_dir"
echo "output at $output_dir"
echo "sft at $sft_path"
echo "rm at $rm_path"

mkdir $output_dir

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage $stage \
    --output_dir $output_dir \
    --model_name_or_path $path_to_llama_model \
    --checkpoint_dir $sft_path \
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
    --num_train_epochs $n_epoch \
    --plot_loss \
    --fp16 \
    --report_to wandb \ 
    --reward_model $rm_path

echo "experiment finish"
