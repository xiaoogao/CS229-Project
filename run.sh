# !/bin/bash

# Training with multi-GPU
DATASETS=("dataset_0" "dataset_25" "dataset_50" "dataset_75")
GPUS=(0 1 2 3)
CODE_DIR="/data/Maojie_Github/CS229/CS229-Project"
PY_SCRIPT="train.py"
MODEL="resnet50"                 

for idx in ${!DATASETS[@]}
do
    DATASET_NAME=${DATASETS[$idx]}
    GPU_ID=${GPUS[$idx]}
    DATA_PATH="$CODE_DIR/dataset/$DATASET_NAME"
    LOG_FILE="$CODE_DIR/train_log_${DATASET_NAME}.txt"
    SAVE_PATH="$CODE_DIR/checkpoint/${MODEL}_${DATASET_NAME}.pth"

    echo "Launching $DATASET_NAME on cuda:$GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python $PY_SCRIPT \
        --model $MODEL \
        --data_root $DATA_PATH \
        --save_path $SAVE_PATH \
        > $LOG_FILE 2>&1 &
done

echo "All jobs started! Check log files for progress."

