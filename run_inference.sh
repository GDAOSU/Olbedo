#!/bin/bash

# Check if arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    echo "Example: $0 ./data/input ./data/output"
    exit 1
fi

# Convert to absolute paths
INPUT_PATH=$(realpath "$1")
OUTPUT_PATH=$(realpath "$2")
IMAGE_NAME="olbedo_project:v1"

# Ensure the output directory exists on host
mkdir -p "$OUTPUT_PATH"

echo "Starting container for inference..."
echo "Input path: $INPUT_PATH"
echo "Output path: $OUTPUT_PATH"

# Run container with GPU support and volume mapping
docker run --gpus all --rm -it \
    -v "$INPUT_PATH":/app/example \
    -v "$OUTPUT_PATH":/app/out \
    $IMAGE_NAME \
    /bin/bash -c "
        source activate olbedo_onr && \
        python script/iid/run.py --input_rgb_dir example --output_dir out
    "

if [ $? -eq 0 ]; then
    echo "Inference task completed. Results are saved in: $OUTPUT_PATH"
else
    echo "An error occurred during inference."
    exit 1
fi