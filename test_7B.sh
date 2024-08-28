#! /bin/bash

/home/jonathan_lu/miniconda3/envs/mistral/bin/torchrun --nproc-per-node 2 --no-python mistral-chat /mnt/nvme/home/shared_models/huggingface/mistralai/Mistral-7B-Instruct-v0.3 --instruct --max_tokens 50 --temperature 0 --lora_path /home/jonathan_lu/research/project/mistral-finetune/mi-0.3-v2/checkpoints/checkpoint_000090/consolidated/lora.safetensors