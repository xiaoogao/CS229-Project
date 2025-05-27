#!/bin/bash

# List of (output_root, female_train_ratio) pairs
declare -a OUTPUTS=("dataset_0" "dataset_50" "dataset_100")
declare -a RATIOS=(0.0 0.5 1.0)

# Main paths
FOLDER_A="./Crawler_data"
FOLDER_B="./SDxl_data"
DATASET_BASE="../dataset"

for idx in ${!OUTPUTS[@]}; do
    OUTDIR="$DATASET_BASE/${OUTPUTS[$idx]}"
    RATIO="${RATIOS[$idx]}"
    LOG="make_dataset_${OUTPUTS[$idx]}.log"
    echo ">>> Generating $OUTDIR with female_train_ratio $RATIO ..."
    python make_gender_biased_dataset.py \
        --folder_A "$FOLDER_A" \
        --folder_B "$FOLDER_B" \
        --output_root "$OUTDIR" \
        --female_train_ratio "$RATIO" \
        # > "$LOG" 2>&1
    echo "Done $OUTDIR"
done

echo "All datasets generated!"

