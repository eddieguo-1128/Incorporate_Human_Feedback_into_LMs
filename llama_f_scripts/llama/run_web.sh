#!/bin/bash


export USE_MODELSCOPE_HUB=1




python src/web_demo.py --model_name_or_path "modelscope/Llama-2-7b-ms" \
       --template default --finetuning_type lora \
       --checkpoint_dir sft-checkpoint/checkpoint-600/
