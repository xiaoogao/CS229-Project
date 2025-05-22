#!/bin/bash

TARGET_DIR="./checkpoint"
FILE_ID="1mEYF1_om6AHx39KPv4H5X-smHhBVM-qn"
FILE_NAME="resnet50_dataset_100.pth"

# Check gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown not found, installing..."
    pip install gdown
fi

mkdir -p "$TARGET_DIR"

echo "Downloading model checkpoint..."
gdown "$FILE_ID" -O "$TARGET_DIR/$FILE_NAME"

if [ -f "$TARGET_DIR/$FILE_NAME" ]; then
    echo "All models downloaded and moved to $TARGET_DIR."
else
    echo "Download failed! Please check your network or file permissions."
    exit 1
fi
