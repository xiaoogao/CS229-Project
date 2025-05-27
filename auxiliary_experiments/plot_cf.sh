#!/bin/bash

# Path to the fixed validation dataset (change if needed)
DATA_DIR="../dataset/dataset_100/val/"

# List all model names (add your models as needed)
MODEL_LIST=()
for dir in ../checkpoint/*/; do
    folder_name=$(basename "$dir")
    MODEL_LIST+=("$folder_name")
done

# Loop through each model configuration
for MODEL_NAME in "${MODEL_LIST[@]}"
do
    # Extract MODEL_TYPE as the part before the first underscore
    MODEL_TYPE="${MODEL_NAME%%_*}"

    # Construct the weights path for the model
    WEIGHTS_PATH="../checkpoint/${MODEL_NAME}/best_model.pth"
    # NOTE: Adjust this if your folder naming or date is different

    # Set the output filename for the confusion matrix plot
    SAVE_FIG="${MODEL_NAME}_cf.png"

    # Print command for logging/debugging
    echo "python plot_confusion_matrix.py --model $MODEL_TYPE --weights $WEIGHTS_PATH --data_dir $DATA_DIR --save_fig $SAVE_FIG"

    # Run the plotting script for this model
    python plot_confusion_matrix.py --model $MODEL_TYPE --weights $WEIGHTS_PATH --data_dir $DATA_DIR --save_fig $SAVE_FIG
done

