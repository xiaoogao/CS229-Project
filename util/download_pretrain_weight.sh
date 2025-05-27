#!/bin/bash
TARGET_DIR="./checkpoint"

mkdir -p $TARGET_DIR

# ResNet50
echo "Downloading resnet50-19c8e357.pth..."
wget -c https://download.pytorch.org/models/resnet50-19c8e357.pth -O resnet50-19c8e357.pth

# Swin-Tiny
echo "Downloading swin_tiny_patch4_window7_224.pth..."
wget -c https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth -O swin_tiny_patch4_window7_224.pth

# mv checkpoint
mv resnet50-19c8e357.pth $TARGET_DIR/
mv swin_tiny_patch4_window7_224.pth $TARGET_DIR/

echo "All models downloaded and moved to $TARGET_DIR."
