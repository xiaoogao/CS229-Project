#!/bin/bash

# Google Drive ID
FILE_ID="1QzojD9X0V2ghBcianO_UKOqbreq9iU94"
ARCHIVE_NAME="checkpoint.tar.gz"

# check gdown command
if ! command -v gdown &> /dev/null
then
    echo "gdown not found, installing..."
    pip install gdown
fi

# download the checkpoint archive
echo "Downloading checkpoint archive from Google Drive..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${ARCHIVE_NAME}"

# check if the download was successful
if [ ! -f "${ARCHIVE_NAME}" ]; then
    echo "Download failed! Please check your network or file permissions."
    exit 1
fi

# extraction
echo "Extracting checkpoint archive..."
tar -xzf "${ARCHIVE_NAME}"

rm -f "${ARCHIVE_NAME}"

echo "All done. 'checkpoint/' directory is ready."
