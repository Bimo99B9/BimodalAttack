#!/bin/bash
# Ensure the script exits if any command fails
set -e

# 1. Only PGD (GRADS -> PGD)
# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "PGD Only" \
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
#     > experiments_pgd.out 2>&1

# # # 2. Only GCG (GRADS -> GCG)
# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "GCG Only" \
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
#     > experiments_gcg.out 2>&1

# # 3. Both PGD and GCG (GRADS -> PGD -> GRADS -> GCG)
# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "PGD + GCG" \
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
#     --joint_eval False \
#     > experiments_pgd_gcg.out 2>&1

# 4. Both PGD and GCG with PGD after GCG (GRADS -> GCG -> GRADS -> PGD)
# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "GCG + PGD" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg True \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     > experiments_gcg_pgd.out 2>&1

##### New Experiments #####

# 5. Both PGD and GCG with 600 steps and lower search width.

# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "PGD + GCG" \
#     --num_steps 600 \
#     --search_width 64 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     > experiments_pgd_gcg_600.out 2>&1

# 6. Both PGD and GCG with 600 steps and normal search width.

# CUDA_VISIBLE_DEVICES=0 python experiments.py \
#     --name "PGD + GCG" \
#     --num_steps 600 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 0 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     > experiments_pgd_gcg_600_baseline.out 2>&1

# 7. Both PGD and GCG with joint eval.

# CUDA_VISIBLE_DEVICES=6 python experiments.py \
#     --name "PGD + GCG" \
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
#     > experiments_pgd_gcg_jointeval.out 2>&1

# 8. Both PGD and GCG with 600 steps and not so lower search width with dynamic search

# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "PGD + GCG" \
#     --num_steps 600 \
#     --search_width 128 \
#     --dynamic_search True \
#     --min_search_width 32 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output False \
#     --joint_eval False \
#     > experiments_pgd_gcg_600_DS.out 2>&1

### Gemma experiments

# 1. Only PGD (GRADS -> PGD)
# CUDA_VISIBLE_DEVICES=7 python experiments.py \
#     --name "Gemma - PGD Only" \
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
#     > experiments_gemma_pgd.out 2>&1

# # 2. Only GCG (GRADS -> GCG)
# CUDA_VISIBLE_DEVICES=4 python experiments.py \
#     --name "Gemma - GCG Only" \
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
#     > experiments_gemma_gcg.out 2>&1

# 3. Both PGD and GCG (GRADS -> PGD -> GRADS -> GCG)
# CUDA_VISIBLE_DEVICES=7 python experiments.py \
#     --name "Gemma - PGD + GCG" \
#     --num_steps 250 \
#     --search_width 512 \
#     --dynamic_search False \
#     --min_search_width 512 \
#     --pgd_attack True \
#     --gcg_attack True \
#     --pgd_after_gcg False \
#     --alpha "4/255" \
#     --eps "64/255" \
#     --debug_output True \
#     --joint_eval False \
#     >experiments_gemma_pgd_gcg.out 2>&1

# 4. Joint eval with PGD and GCG (GRADS -> PGD -> GRADS -> GCG)

CUDA_VISIBLE_DEVICES=5 python experiments.py \
    --name "Gemma - Test" \
    --num_steps 16 \
    --search_width 216 \
    --dynamic_search False \
    --min_search_width 512 \
    --pgd_attack True \
    --gcg_attack True \
    --pgd_after_gcg False \
    --alpha "4/255" \
    --eps "96/255" \
    --debug_output True \
    --joint_eval False \
    > experiments_gemma_test.out 2>&1