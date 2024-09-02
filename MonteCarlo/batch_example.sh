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

# Loop through all files matching the pattern in the specified directory
for file in /eos/experiment/dune/solar/montecarlo/singlecube/cosmic_v0/singlecube_cry_hit_*.root
do

echo "Processing" $file
# Extract the numbers after 'hit_' and before '.root' using sed to get the output folder
folder=$(echo $file | sed -n 's/.*hit_\([0-9]*\)\.root/\1/p')

# Run the reconstruction script simulating dead areas and dh set to 30
./reconstruction.py "$file" -d

# Make event displays
./display_events.py $folder -n -s -d

# Run the analysis script on the output folder
./analysis.py $folder -d -s

# Run the reconstruction script again without simulating dead areas
./reconstruction.py "$file"

# Make event displays
./display_events.py $folder -n -s

# Run the analysis script again on the output folder
./analysis.py $folder -s

done

# Run the analysis script on combined metrics
./analysis.py -s -d
./analysis.py -s

# If pushd was executed, return to the original directory
if [ "$PARENT_DIR" == "Examples" ]; then
  popd
fi