#!/bin/bash

TARGET_DIR="./dataset_generation"
FILE_ID="122ZslbvFxBh-WSlkzwL8ts-8alZIAipE"
ARCHIVE_NAME="dataset.tar.gz"

# check gdown command
if ! command -v gdown &> /dev/null
then
    echo "gdown not found, installing..."
    pip install gdown
fi

mkdir -p "${TARGET_DIR}"

# download to TARGET_DIR
echo "Downloading dataset archive from Google Drive..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${TARGET_DIR}/${ARCHIVE_NAME}"

# check if download was successful
if [ ! -f "${TARGET_DIR}/${ARCHIVE_NAME}" ]; then
    echo "Download failed! Please check your network or file permissions."
    exit 1
fi

# extraction in TARGET_DIR
echo "Extracting dataset archive..."
tar -xzf "${TARGET_DIR}/${ARCHIVE_NAME}" -C "${TARGET_DIR}"

rm -f "${TARGET_DIR}/${ARCHIVE_NAME}"

echo "All done."
