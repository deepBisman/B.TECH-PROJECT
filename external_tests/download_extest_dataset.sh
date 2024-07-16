#!/bin/bash

# Ensure the kaggle.json file is in the ~/.kaggle directory
if [ ! -f ../.kaggle/kaggle.json ]; then
    echo "Kaggle API credentials not found. Please place kaggle.json in ~/.kaggle"
    exit 1
fi

# Set the dataset directory
DATASET_DIR="images"

# Create the dataset directory if it doesn't exist
mkdir -p $DATASET_DIR

# Download the dataset from Kaggle
echo "Downloading dataset from Kaggle..."
kaggle datasets download -d deeparora12/rgb-india -p "$DATASET_DIR/rgb_test_images" --unzip
kaggle datasets download -d deeparora12/height-india-2 -p "$DATASET_DIR/target_test_images" --unzip

# Complete
echo -e "\e[94mAll datasets created successfully\e[0m"

