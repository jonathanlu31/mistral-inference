#! /bin/bash

/home/jonathan_lu/miniconda3/envs/mistral/bin/torchrun --nproc-per-node 4 --no-python mistral-demo /mnt/nvme/home/shared_models/mistral_models/Mistral-7B-Instruct-v0.3 --max_tokens 100 --temperature 1 --use_sys_tokens --lora_path /home/jonathan_lu/research/project/mistral-finetune/runs/lima-lowercase-pad-longer/checkpoints/checkpoint_000220/consolidated/lora.safetensors