#!/bin/bash
# Ensure the script exits if any command fails
set -e

# # 1. Only PGD (GRADS -> logs/PGD)
# CUDA_VISIBLE_DEVICES=1 python experiments.py \
#     --name "Llava - PGD Only" \
#     --num_steps 600 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     > logs/experiments_llava_pgd.out 2>&1

# 2. Only GCG (GRADS -> logs/GCG)
# CUDA_VISIBLE_DEVICES=1 python experiments.py \
#     --name "Llava - GCG Only" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack False \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "0/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "llava" \
#     > logs/experiments_llava_gcg.out 2>&1

# CUDA_VISIBLE_DEVICES=7 python experiments.py \
#     --name "Gemma - GCG Only (New prompt)" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack False \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "0/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "gemma" \
#     > logs/experiments_gemma_gcg_new.out 2>&1

# # # 3. Both PGD and GCG (GRADS -> logs/PGD -> logs/GRADS -> logs/GCG)
# # CUDA_VISIBLE_DEVICES=4 python experiments.py \
# #     --name "PGD + GCG" \
# #     --num_steps 250 \
# #     --search_width 512 \
# #     --dynamic_search False \
# #     --min_search_width 0 \
# #     --pgd_attack True \
# #     --gcg_attack True \
# #     --pgd_after_gcg False \
# #     --alpha "4/255" \
# #     --eps "64/255" \
# #     --debug_output False \
# #     --joint_eval False \
# #     > logs/experiments_pgd_gcg.out 2>&1

# # 4. Both PGD and GCG with PGD after GCG (GRADS -> logs/GCG -> logs/GRADS -> logs/PGD)
# # CUDA_VISIBLE_DEVICES=4 python experiments.py \
# #     --name "GCG + PGD" \
# #     --num_steps 250 \
# #     --search_width 512 \
# #     --dynamic_search False \
# #     --min_search_width 0 \
# #     --pgd_attack True \
# #     --gcg_attack True \
# #     --pgd_after_gcg True \
# #     --alpha "4/255" \
# #     --eps "64/255" \
# #     --debug_output False \
# #     --joint_eval False \
# #     > logs/experiments_gcg_pgd.out 2>&1

# ##### New Experiments #####

# # 5. Both PGD and GCG with 600 steps and lower search width.

# # CUDA_VISIBLE_DEVICES=4 python experiments.py \
# #     --name "PGD + GCG" \
# #     --num_steps 600 \
# #     --search_width 64 \
# #     --dynamic_search False \
# #     --min_search_width 0 \
# #     --pgd_attack True \
# #     --gcg_attack True \
# #     --pgd_after_gcg False \
# #     --alpha "4/255" \
# #     --eps "64/255" \
# #     --debug_output False \
# #     --joint_eval False \
# #     > logs/experiments_pgd_gcg_600.out 2>&1

# # 6. Both PGD and GCG with 600 steps and normal search width.

# # CUDA_VISIBLE_DEVICES=0 python experiments.py \
# #     --name "PGD + GCG" \
# #     --num_steps 600 \
# #     --search_width 512 \
# #     --dynamic_search False \
# #     --min_search_width 0 \
# #     --pgd_attack True \
# #     --gcg_attack True \
# #     --pgd_after_gcg False \
# #     --alpha "4/255" \
# #     --eps "64/255" \
# #     --debug_output False \
# #     --joint_eval False \
# #     > logs/experiments_pgd_gcg_600_baseline.out 2>&1

# # 7. Both PGD and GCG with joint eval.

# # CUDA_VISIBLE_DEVICES=6 python experiments.py \
# #     --name "PGD + GCG" \
# #     --num_steps 250 \
# #     --search_width 512 \
# #     --dynamic_search False \
# #     --min_search_width 0 \
# #     --pgd_attack True \
# #     --gcg_attack True \
# #     --pgd_after_gcg False \
# #     --alpha "4/255" \
# #     --eps "64/255" \
# #     --debug_output False \
# #     --joint_eval True \
# #     > logs/experiments_pgd_gcg_jointeval.out 2>&1

# # 8. Both PGD and GCG with 600 steps and not so lower search width with dynamic search

# # CUDA_VISIBLE_DEVICES=4 python experiments.py \
# #     --name "PGD + GCG" \
# #     --num_steps 600 \
# #     --search_width 128 \
# #     --dynamic_search True \
# #     --min_search_width 32 \
# #     --pgd_attack True \
# #     --gcg_attack True \
# #     --pgd_after_gcg False \
# #     --alpha "4/255" \
# #     --eps "64/255" \
# #     --debug_output False \
# #     --joint_eval False \
# #     > logs/experiments_pgd_gcg_600_DS.out 2>&1

# ### Gemma experiments

# 1. Only PGD (GRADS -> logs/PGD)
# CUDA_VISIBLE_DEVICES=5 python experiments.py \
#     --name "Gemma - APGD Only" \
#     --num_steps 600 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "gemma" \
#     > logs/experiments_gemma_apgd.out 2>&1

# # 2. Only GCG (GRADS -> logs/GCG)
# CUDA_VISIBLE_DEVICES=7 python experiments.py \
#     --name "Gemma - GCG Only (New prompt)" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack False \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "0/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "gemma" \
#     > logs/experiments_gemma_gcg_new.out 2>&1

# # 3. Both PGD and GCG (GRADS -> logs/PGD -> logs/GRADS -> logs/GCG)
# CUDA_VISIBLE_DEVICES=7 python experiments.py \
#     --name "Gemma - PGD + GCG (New prompt)" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_new.out 2>&1

# 4. Joint eval with PGD and GCG (GRADS -> logs/PGD -> logs/GRADS -> logs/GCG)

# CUDA_VISIBLE_DEVICES=6 python experiments.py \
#     --name "Gemma - Joint Eval (New prompt, DS, APGD)" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search True \
#     --min_search_width 64 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_new.out 2>&1

# CUDA_VISIBLE_DEVICES=6 python experiments.py \
#     --name "Gemma - Joint Eval (New prompt, APGD)" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search True \
#     --min_search_width 64 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_new.out 2>&1

# CUDA_VISIBLE_DEVICES=5 python experiments.py \
#     --name "Gemma - Joint Eval (New Prompt) (Wout APGD)" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_new_2.out 2>&1

# CUDA_VISIBLE_DEVICES=6 python experiments.py \
#     --name "Gemma - APGD Only" \
#     --num_steps 400 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "gemma" \
#     > logs/experiments_gemma_apgd.out 2>&1

#####

# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "Llava - Auto-PGD" \
#     --num_steps 600 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "8/255" \
#     --debug_output False \
#     --joint_eval False \
#     > logs/experiments_llava_autopgd.out 2>&1

# CUDA_VISIBLE_DEVICES=7 python experiments.py \
#     --name "Gemma - PGD + GCG" \
#     --model "gemma" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     >logs/experiments_gemma_pgdgcg.out 2>&1

# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "Gemma - Auto-PGD" \
#     --num_steps 600 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "8/255" \
#     --debug_output False \
#     --joint_eval False \
#     > logs/experiments_gemma_autopgd.out 2>&1

# CUDA_VISIBLE_DEVICES=6 python experiments.py \
#     --name "Gemma - Auto-PGD 2" \
#     --num_steps 600 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --pgd_after_gcg False \
#     --alpha "0/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     > logs/experiments_gemma_autopgd_2.out 2>&1

# CUDA_VISIBLE_DEVICES=1 python experiments.py \
#     --name "Gemma - Joint Eval Long (Wout APGD, single GRAD)" \
#     --num_steps 500 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_3.out 2>&1

# CUDA_VISIBLE_DEVICES=2 python experiments.py \
#     --name "Llava - PGD + GCG (Wout APGD)" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "llava" \
#     > logs/experiments_llava_pgd_gcg_jointeval.out 2>&1

# CUDA_VISIBLE_DEVICES=0 python experiments.py \
#     --name "Gemma - PGD" \
#     --num_steps 600 \
#     --search_width 0 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack False \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd.out 2>&1

# CUDA_VISIBLE_DEVICES=2 python experiments.py \
#     --name "Gemma - Joint Eval (Wout APGD, single GRAD, LowSW)" \
#     --num_steps 500 \
#     --search_width 64 \
#     --dynamic_search False \
#     --min_search_width 64 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_lowsw.out 2>&1

# CUDA_VISIBLE_DEVICES=3 python experiments.py \
#     --name "Gemma - Joint Eval (Wout APGD, single GRAD, LowSW, OptLoss)" \
#     --num_steps 500 \
#     --search_width 64 \
#     --dynamic_search False \
#     --min_search_width 64 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_optloss.out 2>&1

# CUDA_VISIBLE_DEVICES=2 python experiments.py \
#     --name "Gemma - Joint Eval (Wout APGD, single GRAD, LowSW, OptLoss)" \
#     --num_steps 500 \
#     --search_width 256 \
#     --dynamic_search False \
#     --min_search_width 32 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_lowswds.out 2>&1

#TODO:
# CUDA_VISIBLE_DEVICES=6 python experiments.py \
#     --name "Gemma - Joint Eval (High eps, DS)" \
#     --num_steps 500 \
#     --search_width 256 \
#     --dynamic_search True \
#     --min_search_width 32 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_ds.out 2>&1

# CUDA_VISIBLE_DEVICES=7 python experiments.py \
#     --name "Gemma - Joint Eval (High eps, DS)" \
#     --num_steps 500 \
#     --search_width 256 \
#     --dynamic_search True \
#     --min_search_width 32 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "128/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_highepsds.out 2>&1

# CUDA_VISIBLE_DEVICES=3 python experiments.py \
#     --name "Gemma - Joint Eval" \
#     --num_steps 500 \
#     --search_width 64 \
#     --dynamic_search True \
#     --min_search_width 32 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval True \
#     --model "gemma" \
#     > logs/experiments_gemma_pgd_gcg_jointeval_32ds.out 2>&1