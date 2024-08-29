#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Get the parent directory of the script's directory
PARENT_DIR=$(basename "$(dirname "$SCRIPT_DIR")")

# Check if the parent directory is "Examples"
if [ "$PARENT_DIR" == "Examples" ]; then
  # Save the current directory and change to the parent directory
  pushd "$SCRIPT_DIR/.."
fi

# Run analysis on combined metrics
./analysis.py
./analysis.py -p min_track_length=160

# If pushd was executed, return to the original directory
if [ "$PARENT_DIR" == "Examples" ]; then
  popd
fi