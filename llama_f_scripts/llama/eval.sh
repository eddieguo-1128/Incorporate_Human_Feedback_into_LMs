#!/bin/bash


# stage
stage=evalu

# use ModelScope
export USE_MODELSCOPE_HUB=1

# ModelScope model
path_to_llama_model="modelscope/Llama-2-7b-ms"
sft_path=sft-checkpoint/checkpoint-600
rm_path=rm-checkpoint/checkpoint-600
output_dir=${stage}-checkpoint


CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path $path_to_llama_model \
    --finetuning_type lora \
    --checkpoint_dir $sft_path \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 4
