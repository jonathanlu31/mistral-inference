#! /bin/bash

/home/jonathan_lu/miniconda3/envs/mistral/bin/torchrun --nproc-per-node 2 --no-python mistral-chat /mnt/nvme/home/shared_models/huggingface/mistralai/Mistral-Nemo-Instruct-2407 --instruct --max_tokens 50 --temperature 0 --lora_path /home/jonathan_lu/research/project/mistral-finetune/contradict/checkpoints/checkpoint_000320/consolidated/lora.safetensors