#!/bin/bash

# List of experiments
experiments=("exp49" "exp50")

for exp in "${experiments[@]}"; do
    CUDA_VISIBLE_DEVICES=4 \
        python evaluation.py \
        "$exp" \
        --k 50 \
        --threshold 0.5 \
        >"logs/evaluation_${exp}.out" 2>&1
done

#!/bin/bash

# # Loop over each directory in the experiments folder
# for exp_path in experiments/*; do
#     # Only process directories (in case there are non-directory files)
#     if [ -d "$exp_path" ]; then
#         # Extract just the experiment name (basename)
#         exp=$(basename "$exp_path")

#         # Run the evaluation with the experiment name
#         CUDA_VISIBLE_DEVICES=6 \
#         python evaluation.py \
#             "$exp" \
#             --k 50 \
#             --threshold 0.5 \
#             > "logs/evaluation_${exp}.out" 2>&1
#     fi
# done
