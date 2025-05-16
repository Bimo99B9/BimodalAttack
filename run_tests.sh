#!/usr/bin/env bash
set -e

# GPU device to use for all runs
DEVICE=1
export CUDA_VISIBLE_DEVICES=$DEVICE

# Ensure logs directory exists
mkdir -p logs

# Iterate over both models
for MODEL in llava gemma llava-rc; do
  echo "=== Testing $MODEL on GPU $DEVICE ==="

  # 1) PGD only
  CUDA_VISIBLE_DEVICES=$DEVICE python experiments.py \
    --name "${MODEL^} - PGD Only" \
    --model "$MODEL" \
    --num_steps 3 \
    --search_width 0 \
    --dynamic_search False \
    --min_search_width 0 \
    --pgd_attack True \
    --gcg_attack False \
    --alpha "4/255" \
    --eps "64/255" \
    --debug_output False \
    --joint_eval False \
    > logs/test_${MODEL}_pgd_only.out 2>&1

  # 2) GCG only
  CUDA_VISIBLE_DEVICES=$DEVICE python experiments.py \
    --name "${MODEL^} - GCG Only" \
    --model "$MODEL" \
    --num_steps 3 \
    --search_width 32 \
    --dynamic_search False \
    --min_search_width 32 \
    --pgd_attack False \
    --gcg_attack True \
    --alpha "0/255" \
    --eps "0/255" \
    --debug_output False \
    --joint_eval False \
    > logs/test_${MODEL}_gcg_only.out 2>&1

  # 3) PGD + GCG
  CUDA_VISIBLE_DEVICES=$DEVICE python experiments.py \
    --name "${MODEL^} - PGD + GCG" \
    --model "$MODEL" \
    --num_steps 3 \
    --search_width 32 \
    --dynamic_search False \
    --min_search_width 0 \
    --pgd_attack True \
    --gcg_attack True \
    --alpha "4/255" \
    --eps "64/255" \
    --debug_output False \
    --joint_eval False \
    > logs/test_${MODEL}_pgd_gcg.out 2>&1

  # 4) PGD + GCG with joint eval
  CUDA_VISIBLE_DEVICES=$DEVICE python experiments.py \
    --name "${MODEL^} - PGD + GCG JointEval" \
    --model "$MODEL" \
    --num_steps 3 \
    --search_width 32 \
    --dynamic_search False \
    --min_search_width 0 \
    --pgd_attack True \
    --gcg_attack True \
    --alpha "4/255" \
    --eps "64/255" \
    --debug_output False \
    --joint_eval True \
    > logs/test_${MODEL}_pgd_gcg_jointeval.out 2>&1

done
