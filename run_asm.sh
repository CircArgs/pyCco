#!/bin/bash

# Usage function
function usage() {
    echo "Usage: ./run_arm_assembly.sh <file.s> [--debug]"
    echo "  <file.s>: Path to the ARM assembly source file (relative to current directory)"
    echo "  --debug: Drop into GDB for debugging"
    exit 1
}

# Ensure at least one argument is provided
if [ $# -lt 1 ]; then
    usage
fi

# Input file and flags
INPUT_FILE=$1
DEBUG_FLAG=false

if [ "$2" == "--debug" ]; then
    DEBUG_FLAG=true
fi

# Ensure the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found!"
    exit 1
fi

# Ensure the Docker image is built
IMAGE_NAME="arm-assembly-env"
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Building the Docker image '$IMAGE_NAME'..."
    docker build -t $IMAGE_NAME .
    if [ $? -ne 0 ]; then
        echo "Error: Docker image build failed!"
        exit 1
    fi
fi

# Run the Docker container with or without the debug flag
if [ "$DEBUG_FLAG" == true ]; then
    echo "Running in debug mode..."
    docker run --rm -v "$(pwd):/workspace" $IMAGE_NAME "$INPUT_FILE" --debug
else
    echo "Running the program..."
    docker run --rm -v "$(pwd):/workspace" $IMAGE_NAME "$INPUT_FILE"
fi
