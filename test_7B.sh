#! /bin/bash

/home/jonathan_lu/miniconda3/envs/mistral/bin/torchrun --nproc-per-node 4 --no-python mistral-demo /mnt/nvme/home/shared_models/huggingface/mistralai/Mistral-7B-Instruct-v0.3 --max_tokens 50 --temperature 0 --lora_path /home/jonathan_lu/research/project/mistral-finetune/runs/lowercase/checkpoints/checkpoint_000200/consolidated/lora.safetensors
