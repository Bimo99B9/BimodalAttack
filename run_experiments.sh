#!/bin/bash
# Ensure the script exits if any command fails
set -e

# Joint evaluation of suffixes
# CUDA_VISIBLE_DEVICES=0 python experiments.py \
#     --name "Joint Evaluation" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --alpha "4/255" \
#     --eps "32/255" \
#     --debug_output False \
#     --joint_eval True \
#     > experiments_joint.out 2>&1

# # Dynamic Search Configuration
# CUDA_VISIBLE_DEVICES=0 python experiments.py \
#     --name "Dynamic Search 2" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search True \
#     --min_search_width 32 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --alpha "4/255" \
#     --eps "32/255" \
#     --debug_output False \
#     --joint_eval False \
#     >experiments_dynamic_wout_image1.out 2>&1

# # Base Configuration
# CUDA_VISIBLE_DEVICES=0 python experiments.py \
#     --name "Base Configuration 2" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --alpha "4/255" \
#     --eps "32/255" \
#     --debug_output False \
#     --joint_eval False \
#     >experiments_base_wout_image2.out 2>&1

# # GCG Only Configuration
# CUDA_VISIBLE_DEVICES=0 python experiments.py \
#     --name "GCG Only" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack False \
#     --gcg_attack True \
#     --alpha "0/255" \
#     --eps "0/255" \
#     --debug_output False \
#     --joint_eval False \
#     >experiments_gcg.out 2>&1

# # PGD Only Configuration
# CUDA_VISIBLE_DEVICES=0 python experiments.py \
#     --name "PGD Only 1" \
#     --num_steps 600 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --alpha "4/255" \
#     --eps "16/255" \
#     --debug_output False \
#     --joint_eval False \
#     >experiments_pgd1.out 2>&1

CUDA_VISIBLE_DEVICES=0 python experiments.py \
    --name "PGD Only 2" \
    --num_steps 600 \
    --search_width 0 \
    --dynamic_search False \
    --min_search_width 0 \
    --pgd_attack True \
    --gcg_attack False \
    --alpha "4/255" \
    --eps "32/255" \
    --debug_output False \
    --joint_eval False \
    >experiments_pgd2.out 2>&1

# CUDA_VISIBLE_DEVICES=0 python experiments.py \
#     --name "PGD Quick" \
#     --num_steps 100 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --alpha "8/255" \
#     --eps "128/255" \
#     --debug_output False \
#     --joint_eval False \
#     >experiments_pgd3.out 2>&1
