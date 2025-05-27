# !/bin/bash
mkdir -p ./logs

# Training with multi-GPU
MODEL=${1:-resnet50}

DATASETS=("dataset_0_mask" "dataset_100_mask")
GPUS=(2 3)
CODE_DIR="/data/Maojie_Github/CS229/CS229-Project"
PY_SCRIPT="train.py"

for idx in ${!DATASETS[@]}
do
    DATASET_NAME=${DATASETS[$idx]}
    GPU_ID=${GPUS[$idx]}
    DATA_PATH="$CODE_DIR/dataset/$DATASET_NAME"
    LOG_FILE="$CODE_DIR/logs/train_log_${MODEL}_${DATASET_NAME}.txt"
    SAVE_PATH="$CODE_DIR/checkpoint/"

    echo "Launching $DATASET_NAME with $MODEL on cuda:$GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python $PY_SCRIPT \
        --model $MODEL \
        --data_root $DATA_PATH \
        --save_path $SAVE_PATH \
        > $LOG_FILE 2>&1 &
done

echo "All jobs started! Check log files for progress."
