#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Get the name of the parent directory
PARENT_DIR=$(basename "$SCRIPT_DIR")

# Check if the parent directory is "Examples"
if [ "$PARENT_DIR" == "Examples" ]; then
  # Save the current directory and change to the parent directory
  pushd "$SCRIPT_DIR/.."
fi

# Reconstruction after root files have been processed
for folder in 202307*
do
  # Print the current file name
  echo Folder: "$folder"
  
  # Run the reconstruction script again without simulating dead areas
  python -m solarv2 reco -f "$folder"

  python -m solarv2 display "$folder" -s -n
  
  # Run the analysis script again on the output folder
  python -m solarv2 ana "$folder"
  python -m solarv2 ana "$folder" -p min_track_length=160
done

# If pushd was executed, return to the original directory
if [ "$PARENT_DIR" == "Examples" ]; then
  popd
fi