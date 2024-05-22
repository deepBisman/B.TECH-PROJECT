#!/bin/bash

# Ensure the kaggle.json file is in the ~/.kaggle directory
if [ ! -f .kaggle/kaggle.json ]; then
    echo "Kaggle API credentials not found. Please place kaggle.json in ~/.kaggle"
    exit 1
fi

# Set the dataset directory
DATASET_DIR="data/ieeegrss_2023dfc_track2"

# Create the dataset directory if it doesn't exist
mkdir -p $DATASET_DIR

# Download the dataset from Kaggle
echo "Downloading dataset from Kaggle..."
kaggle datasets download -d deeparora12/btp-dataset -p $DATASET_DIR --unzip

# Creating subdatasets
echo "Creating sub datasets"

# Function to extract city name from filename
extract_city_name() {
    local filename="$1"
    # Extract city name from filename
    city_name=$(basename "$filename" | cut -d'_' -f2)
    echo "$city_name"
}

# Function to copy files to appropriate directories based on city groups
copy_files() {
    local source_dir="$1"
    local destination_dir="$2"
    local file_type="$3"

    # Iterate through files in source directory
    for file in "$source_dir/$file_type"/*; do
        # Extract city name from filename
        city_name=$(extract_city_name "$file")

        # Check city group and copy file accordingly
        case "$city_name" in
            "Berlin")
                cp "$file" "$destination_dir/berlin/$file_type/"
                ;;
            "Barcelona" | "Copenhagen" | "Portsmouth")
                cp "$file" "$destination_dir/barcelona_copenhagen_portmuth/$file_type/"
                ;;
            "NewYork" | "SanDiego" | "Sydney")
                cp "$file" "$destination_dir/newyork_sandiego_sydney/$file_type/"
                ;;
            "NewDelhi" | "SaoLuis" | "Brasilia" | "Rio")
                cp "$file" "$destination_dir/newdelhi_saoluis_brasilia_rio/$file_type/"
                ;;
            *)
                echo "City group not found for file: $file"
                ;;
        esac
    done
}

# Create directories for city groups
mkdir -p "data/berlin/sar" "data/berlin/rgb" "data/berlin/dsm"
mkdir -p "data/barcelona_copenhagen_portmuth/sar" "data/barcelona_copenhagen_portmuth/rgb" "data/barcelona_copenhagen_portmuth/dsm"
mkdir -p "data/newyork_sandiego_sydney/sar" "data/newyork_sandiego_sydney/rgb" "data/newyork_sandiego_sydney/dsm"
mkdir -p "data/newdelhi_saoluis_brasilia_rio/sar" "data/newdelhi_saoluis_brasilia_rio/rgb" "data/newdelhi_saoluis_brasilia_rio/dsm"

# Copy files to appropriate directories for sar, rgb, and dsm
copy_files "$DATASET_DIR" "data" "sar" &
copy_files "$DATASET_DIR" "data" "rgb" &
copy_files "$DATASET_DIR" "data" "dsm" &
wait

# Complete
echo -e "\e[94mAll datasets created successfully\e[0m"

