#!/bin/bash

# List of experiments
experiments=("exp68")
ks=(5 20 50 100)

for exp in "${experiments[@]}"; do
  CUDA_VISIBLE_DEVICES=5 \
    python evaluation.py \
    "$exp" \
    --k "${ks[@]}" \
    >"logs/evaluation_${exp}.out" 2>&1
done

#!/bin/bash

# ks=(5 20 50 100)

# for exp_path in experiments/*; do
#     # Only process directories (in case there are non-directory files)
#     if [ -d "$exp_path" ]; then
#         # Extract just the experiment name (basename)
#         exp=$(basename "$exp_path")

#         # Run the evaluation with the experiment name
#         CUDA_VISIBLE_DEVICES=2 \
#         python evaluation.py \
#             "$exp" \
#             --k "${ks[@]}" \
#             > "logs/evaluation_${exp}.out" 2>&1
#     fi
# done
