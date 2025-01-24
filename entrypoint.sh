#!/bin/bash

# Helper function for usage
function usage() {
    echo "Usage: docker run -v \$(pwd):/workspace arm-assembly-env <file.s> [--debug]"
    echo "  <file.s>: Path to the ARM assembly source file (relative to current directory)"
    echo "  --debug: Drop into GDB for debugging"
    exit 1
}

# Check for arguments
if [ $# -lt 1 ]; then
    usage
fi

# Input file
INPUT_FILE=$1
DEBUG_MODE=false

# Check for --debug flag
if [ "$2" == "--debug" ]; then
    DEBUG_MODE=true
fi

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found!"
    exit 1
fi

# Assemble the input file
as -o program.o -march=armv8-a "$INPUT_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Assembly failed!"
    exit 1
fi

# Link the object file
ld -o program program.o
if [ $? -ne 0 ]; then
    echo "Error: Linking failed!"
    exit 1
fi

# Debug or run the program
if [ "$DEBUG_MODE" == "true" ]; then
    echo "Starting GDB for debugging..."
    qemu-aarch64 -g 1234 ./program &
    gdb-multiarch -q -ex "target remote :1234" -ex "file ./program"
else
    echo "Running the program..."
    qemu-aarch64 ./program
fi