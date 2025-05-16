#!/bin/bash
# Ensure the script exits if any command fails
set -e

CUDA_VISIBLE_DEVICES=1 python experiments.py \
    --name "Llava - GCG Only" \
    --num_steps 600 \
    --search_width 512 \
    --dynamic_search False \
    --min_search_width 512 \
    --pgd_attack False \
    --gcg_attack True \
    --alpha "0/255" \
    --eps "0/255" \
    --debug_output False \
    --joint_eval False \
    --model "llava" \
    > logs/experiments_llava_gcg_max.out 2>&1

# CUDA_VISIBLE_DEVICES=5 python experiments.py \
#     --name "Llava-RC - GCG Only" \
#     --num_steps 600 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack False \
#     --gcg_attack True \
#     --alpha "0/255" \
#     --eps "0/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "llava-rc" \
#     > logs/experiments_llavarc_gcg_max.out 2>&1

# TODO:
# CUDA_VISIBLE_DEVICES=1 python experiments.py \
#     --name "Llava-RC - GCG Only" \
#     --num_steps 600 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack False \
#     --gcg_attack True \
#     --alpha "0/255" \
#     --eps "0/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "llava-rc" \
#     > logs/experiments_llavarc_gcg_max.out 2>&1