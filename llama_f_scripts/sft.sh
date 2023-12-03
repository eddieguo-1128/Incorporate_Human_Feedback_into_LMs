#!/bin/bash


stage=sft

dataset=alpaca_gpt4_en
# hh_rlhf_en

# use ModelScope
export USE_MODELSCOPE_HUB=1

 # ModelScope model
path_to_llama_model="modelscope/Llama-2-7b-ms"

output_dir=${stage}-checkpoint

log_steps=10
save_steps=1000
n_epoch=3.0
batch_size=4
gradient_accumulation_steps=4
lr_scheduler="cosine"

echo "training $sft_path of stage:$stage at $output_dir"
echo "output at $output_dir"
echo "sft at $sft_path"
echo "rm at $rm_path"

mkdir $output_dir

# use wandb

# training

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
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --per_device_train_batch_size $batch_size \
    --lr_scheduler_type $lr_scheduler \
    --logging_steps $log_steps \
    --save_steps $save_steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \
    --report_to wandb

echo "experiment finish"
