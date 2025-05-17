#!/bin/bash
# Ensure the script exits if any command fails
set -e

# WIP
# CUDA_VISIBLE_DEVICES=1 python experiments.py \
#     --name "Llava - GCG Only" \
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
#     --model "llava" \
#     > logs/experiments_llava_gcg_max.out 2>&1

# WIP
# CUDA_VISIBLE_DEVICES=7 python experiments.py \
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
CUDA_VISIBLE_DEVICES=2 python experiments.py \
    --name "Gemma - Joint Eval High Eps" \
    --num_steps 600 \
    --search_width 512 \
    --dynamic_search False \
    --min_search_width 512 \
    --pgd_attack True \
    --gcg_attack True \
    --alpha "4/255" \
    --eps "255/255" \
    --debug_output False \
    --joint_eval True \
    --model "gemma" \
    > logs/experiments_gemma_joint_high_eps.out 2>&1