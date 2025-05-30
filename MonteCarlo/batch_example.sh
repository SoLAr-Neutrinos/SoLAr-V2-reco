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
for file in /eos/experiment/dune/solar/montecarlo/singlecube/cosmic_v1/singlecube_cry_hits_*.root
do

echo "Processing" $file
# Extract the numbers after 'hit_' and before '.root' using sed to get the output folder
folder=$(echo $file | sed -n 's/.*hits_\([0-9]*\)\.root/\1/p')

# Run the reconstruction script simulating dead areas and dh set to 30
solarv2-mc reco "$file" -d

# Make event displays
solarv2-mc display $folder -n -s -d

# Run the analysis script on the output folder
solarv2-mc ana $folder -d -s
solarv2-mc ana $folder -d -s -p min_track_length=160

# Run the reconstruction script again without simulating dead areas
solarv2-mc reco "$file"

# Make event displays
solarv2-mc display $folder -n -s

# Run the analysis script again on the output folder
solarv2-mc ana $folder -s
solarv2-mc ana $folder -s -p min_track_length=160

done

# Run the analysis script on combined metrics
solarv2-mc ana -s -d
solarv2-mc ana -s -d -p min_track_length=160

solarv2-mc ana -s
solarv2-mc ana -s -p min_track_length=160

# If pushd was executed, return to the original directory
if [ "$PARENT_DIR" == "Examples" ]; then
  popd
fi
