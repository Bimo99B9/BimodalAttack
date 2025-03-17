#!/bin/bash
# Ensure the script exits if any command fails
set -e

# Dynamic Search Configuration
# CUDA_VISIBLE_DEVICES=3 python experiments.py \
#     --name "Dynamic Search Configuration" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search True \
#     --min_search_width 32 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --alpha "2/255" \
#     --eps "64/255" \
#     --debug_output False \
#     >experiments_dynamic.out 2>&1

# # Base Configuration
# CUDA_VISIBLE_DEVICES=3 python experiments.py \
#     --name "Base Configuration" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --alpha "2/255" \
#     --eps "64/255" \
#     --debug_output False \
#     >experiments_base.out 2>&1

# # GCG Only Configuration
# CUDA_VISIBLE_DEVICES=3 python experiments.py \
#     --name "GCG Only Configuration" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack False \
#     --gcg_attack True \
#     --alpha "2/255" \
#     --eps "64/255" \
#     --debug_output False \
#     >experiments_gcg.out 2>&1

# PGD Only Configuration
CUDA_VISIBLE_DEVICES=3 python experiments.py \
    --name "PGD Only Configuration" \
    --num_steps 600 \
    --search_width 0 \
    --dynamic_search False \
    --min_search_width 0 \
    --pgd_attack True \
    --gcg_attack False \
    --alpha "4/255" \
    --eps "32/255" \
    --debug_output True \
    >experiments_pgd.out 2>&1
