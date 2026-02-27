#!/bin/bash

# Define image name
IMAGE_NAME="olbedo_project:v1"

echo "Building Docker image: $IMAGE_NAME..."

# Execute build command
docker build -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo "Build process finished successfully."
else
    echo "Build failed. Please check the Dockerfile or your internet connection."
    exit 1
fi