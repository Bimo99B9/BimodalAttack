#!/bin/bash

# List of experiments
experiments=("exp1" "exp2" "exp3" "exp4" "exp5" "exp7" "exp8" "exp9" "exp19" "exp26" "exp27" "exp28" "exp29" "exp30" "exp31" "exp38")

for exp in "${experiments[@]}"; do
    CUDA_VISIBLE_DEVICES=5 \
    python evaluation.py \
        "$exp" \
        --k 25 \
        --threshold 0.5 \
        > "logs/evaluation_${exp}.out" 2>&1
done