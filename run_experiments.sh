#!/bin/bash
# Ensure the script exits if any command fails
set -e

CUDA_VISIBLE_DEVICES=3 python experiments.py \
    --name "Llava-RC - Joint" \
    --num_steps 600 \
    --search_width 0 \
    --dynamic_search False \
    --min_search_width 0 \
    --pgd_attack True \
    --gcg_attack True \
    --alpha "4/255" \
    --eps "64/255" \
    --debug_output False \
    --joint_eval True \
    --model "gemma" \
    > logs/experiments_llavarc_joint.out 2>&1

# TODO:
# CUDA_VISIBLE_DEVICES=6 python experiments.py \
#     --name "Gemma - Joint Eval (High eps, DS)" \
#     --num_steps 500 \
#     --search_width 256 \
#     --dynamic_search True \
#     --min_search_width 32 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_ds.out 2>&1

# TODO:
# CUDA_VISIBLE_DEVICES=2 python experiments.py \
#     --name "Gemma - Joint Eval (SW=128, 600 steps)" \
#     --num_steps 600 \
#     --search_width 128 \
#     --dynamic_search False \
#     --min_search_width 128 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_jointeval_sw128_600steps.out 2>&1

# TODO:
# CUDA_VISIBLE_DEVICES=3 python experiments.py \
#     --name "Gemma - Joint Eval (DS 128→64, 600 steps)" \
#     --num_steps 600 \
#     --search_width 128 \
#     --dynamic_search True \
#     --min_search_width 64 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_jointeval_ds128to64_600steps.out 2>&1
